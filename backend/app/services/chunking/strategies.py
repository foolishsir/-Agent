"""可选的替代分块策略: 递归字符切分 与 固定长度切分.

这两个策略的存在意义
--------------------
不是为了"更好", 而是为了**有可比对象**:

- ``recursive``: 通用方案, 不依赖标题识别, 对结构不规整的文档更稳
- ``fixed``: 完全不做语义考量的基线

做 RAG 优化时最大的陷阱是"感觉好像好一点". 只有把基线跑出来,
才能说清楚"父子块策略让 Recall@5 从 X 提到 Y" —— 这是简历上能写、
面试官会追问的数字. 没有基线的优化故事是站不住的.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models.document import ChunkType
from app.services.chunking.base import (
    Chunk,
    ChunkingResult,
    coalesce_short_chunks,
    make_child_id,
    make_parent_id,
)
from app.services.chunking.params import ChunkParams, split_sentences
from app.services.parser.base import CleanDocument


@dataclass
class FlatText:
    """把文档拉平成一个字符流, 并保留"每个字符来自哪一页"的映射.

    为什么需要它: 固定长度切分和递归切分都是**按字符位置**切的,
    它们不关心段落边界. 但引用溯源要求每个块能报出准确页码,
    所以必须有一个"字符偏移 → 页码"的索引.

    实现上用一个与文本等长的页码数组: 空间开销可接受(一个 int 列表),
    查询是 O(1) 的. 相比"记录稀疏区间再二分查找", 这种朴素做法更简单也更不容易错.
    """

    text: str
    pages: list[int]

    @classmethod
    def from_document(cls, document: CleanDocument) -> FlatText:
        parts: list[str] = []
        pages: list[int] = []

        for index, paragraph in enumerate(document.paragraphs):
            if index:
                # 段落之间用空行连接 —— 这是 recursive 策略最先尝试的分隔符
                parts.append("\n\n")
                pages.extend([paragraph.page_no] * 2)

            parts.append(paragraph.text)
            pages.extend([paragraph.page_no] * len(paragraph.text))

        return cls(text="".join(parts), pages=pages)

    def page_range(self, start: int, end: int) -> tuple[int, int]:
        """返回字符区间 [start, end) 覆盖的页码范围."""
        if not self.pages:
            return 1, 1
        start = max(0, min(start, len(self.pages) - 1))
        end = max(start + 1, min(end, len(self.pages)))
        window = self.pages[start:end]
        if not window:
            return 1, 1
        return min(window), max(window)

    def slice(self, start: int, end: int) -> str:
        return self.text[start:end]

    def __len__(self) -> int:
        return len(self.text)


# --------------------------------------------------------------------------- #
# 固定长度切分(基线)
# --------------------------------------------------------------------------- #
def chunk_fixed(document: CleanDocument, doc_id: str, params: ChunkParams) -> ChunkingResult:
    """按固定字符数硬切, 带重叠.

    这是**刻意不做任何语义考量**的基线实现:
    不考虑段落、句子、标题, 纯粹按字符数切. 除非有人明确要复现这个基线,
    否则不应该在生产中使用 —— 它会把一句话、一个表格、一个型号从中间切断.
    """
    flat = FlatText.from_document(document)
    spans = _fixed_spans(len(flat), params.child_size, params.overlap)

    return _build_from_spans(flat, spans, doc_id, params, section_path=None)


# --------------------------------------------------------------------------- #
# 递归字符切分
# --------------------------------------------------------------------------- #
def chunk_recursive(document: CleanDocument, doc_id: str, params: ChunkParams) -> ChunkingResult:
    """按分隔符优先级递归切分.

    算法(与 LangChain 的 RecursiveCharacterTextSplitter 同思路)::

        1. 用当前优先级的第一个分隔符把文本切开
        2. 把切出来的片段**尽可能合并**到接近 chunk_size
        3. 若某个片段本身就超过 chunk_size, 换下一个更细的分隔符对它递归
        4. 分隔符全部用尽还超长 → 硬切字符(最后手段)

    这样做的效果是"尽量在语义边界处断开": 优先在段落断开,
    其次换行, 再次句号, 最后才动逗号和字符.
    """
    flat = FlatText.from_document(document)
    # 重叠只在**最外层**统一应用一次.
    # 如果每一层递归都加重叠, 嵌套递归会导致同一个字符被反复计入多个块,
    # 块与块高度冗余 —— 这是递归切分很容易写错的地方.
    spans = _recursive_spans(flat, 0, len(flat), params.separators, params.child_size)
    if not spans:
        spans = _fixed_spans(len(flat), params.child_size, params.overlap)
    else:
        spans = _apply_overlap(spans, params.overlap)

    return _build_from_spans(flat, spans, doc_id, params, section_path=None)


# --------------------------------------------------------------------------- #
# 切分算法
# --------------------------------------------------------------------------- #
def _fixed_spans(total: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """按固定长度切片, 带重叠."""
    if total <= 0:
        return []

    step = max(1, size - overlap)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(start + size, total)
        spans.append((start, end))
        if end >= total:
            break
        start += step
    return spans


def _recursive_spans(
    flat: FlatText,
    start: int,
    end: int,
    separators: list[str],
    size: int,
) -> list[tuple[int, int]]:
    """递归切分, 返回字符偏移区间列表.

    注意这里**不加重叠** —— 重叠由最外层统一应用一次(见 ``chunk_recursive``).
    """
    if end - start <= size:
        return [(start, end)] if end > start else []

    if not separators:
        # 分隔符用尽 → 硬切(最后手段)
        return [(start + a, start + b) for a, b in _fixed_spans(end - start, size, overlap=0)]

    separator, rest = separators[0], separators[1:]
    pieces = _split_keep_separator(flat.text, start, end, separator)

    if len(pieces) <= 1:
        # 当前分隔符在这个区间里不存在 → 换下一个
        return _recursive_spans(flat, start, end, rest, size)

    spans: list[tuple[int, int]] = []
    buffer_start: int | None = None
    buffer_end = 0

    for piece_start, piece_end in pieces:
        piece_len = piece_end - piece_start
        if piece_len > size:
            # 这一段本身就超长 → 先冲掉已累积的, 再对它递归
            if buffer_start is not None:
                spans.append((buffer_start, buffer_end))
                buffer_start = None
            spans.extend(_recursive_spans(flat, piece_start, piece_end, rest, size))
            continue

        if buffer_start is None:
            buffer_start, buffer_end = piece_start, piece_end
        elif buffer_end - buffer_start + piece_len <= size:
            buffer_end = piece_end
        else:
            spans.append((buffer_start, buffer_end))
            buffer_start, buffer_end = piece_start, piece_end

        if buffer_start is not None:
            spans.append((buffer_start, buffer_end))
            buffer_start = None

    return spans


def _split_keep_separator(text: str, start: int, end: int, separator: str) -> list[tuple[int, int]]:
    """按分隔符切分, **把分隔符保留在前一个片段末尾**.

    保留分隔符很重要: 丢掉句号会让句子在展示时失去断句,
    而且在中文里 "。" 是语义边界的一部分, 丢掉会让分块读起来很怪.
    """
    if not separator:
        return [(start, end)]

    pieces: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        found = text.find(separator, cursor, end)
        if found < 0:
            if cursor < end:
                pieces.append((cursor, end))
            break
        pieces.append((cursor, found + len(separator)))
        cursor = found + len(separator)

    return [(s, e) for s, e in pieces if e > s]


def _apply_overlap(
    spans: list[tuple[int, int]], overlap: int, max_carry_ratio: float = 0.5
) -> list[tuple[int, int]]:
    """给相邻区间加上重叠.

    做法是把后一个块的起点向前挪 ``overlap`` 个字符.
    上限 ``max_carry_ratio`` 防止 overlap 配得过大时块与块高度冗余
    (那会让检索结果里出现大量近似重复, 反而降低上下文多样性).
    """
    if overlap <= 0 or len(spans) <= 1:
        return spans

    sizes = [e - s for s, e in spans]
    limit = int(min(sizes) * max_carry_ratio)
    effective = min(overlap, limit) if limit > 0 else 0
    if effective <= 0:
        return spans

    adjusted: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        prev_start = adjusted[-1][0]
        new_start = max(prev_start + 1, start - effective)
        # 不能越过前一个块的起点, 否则会出现起点倒退的无效区间
        new_start = min(new_start, start)
        adjusted.append((new_start, end))
    return adjusted


# --------------------------------------------------------------------------- #
# 由切分区间构造分块对象
# --------------------------------------------------------------------------- #
def _build_from_spans(
    flat: FlatText,
    spans: list[tuple[int, int]],
    doc_id: str,
    params: ChunkParams,
    section_path: str | None,
) -> ChunkingResult:
    """把字符区间转换成父子块结构.

    即使 fixed / recursive 本身不区分层级, 也会把连续子块按大小聚合成父块 ——
    这样三种策略产出的结构完全一致, 下游检索链路不需要写分支.
    """
    # ---------------- 子块 ----------------
    children: list[Chunk] = []
    for index, (start, end) in enumerate(spans):
        text = flat.slice(start, end).strip()
        if not text:
            continue
        page_start, page_end = flat.page_range(start, end)
        children.append(
            Chunk(
                id=make_child_id(doc_id, 0, index),
                doc_id=doc_id,
                parent_id=None,  # 下面分组时再补
                chunk_type=ChunkType.CHILD,
                content=text,
                page_start=page_start,
                page_end=page_end,
                section_path=section_path,
                order_index=index,
            )
        )

    children = _coalesce_short(children, params.min_size)

    # ---------------- 父块: 按大小聚合连续子块 ----------------
    parents: list[Chunk] = []
    groups: list[list[Chunk]] = []
    current: list[Chunk] = []
    current_len = 0

    for child in children:
        if current and current_len + child.char_count > params.parent_size:
            groups.append(current)
            current, current_len = [], 0
        current.append(child)
        current_len += child.char_count

    if current:
        groups.append(current)

    rebuilt_children: list[Chunk] = []
    for parent_index, group in enumerate(groups):
        parent_id = make_parent_id(doc_id, parent_index)
        parents.append(
            Chunk(
                id=parent_id,
                doc_id=doc_id,
                parent_id=None,
                chunk_type=ChunkType.PARENT,
                content="\n\n".join(c.content for c in group),
                page_start=min(c.page_start for c in group),
                page_end=max(c.page_end for c in group),
                section_path=section_path,
                order_index=parent_index,
            )
        )
        for child_index, child in enumerate(group):
            rebuilt_children.append(
                Chunk(
                    id=make_child_id(doc_id, parent_index, child_index),
                    doc_id=doc_id,
                    parent_id=parent_id,
                    chunk_type=ChunkType.CHILD,
                    content=child.content,
                    page_start=child.page_start,
                    page_end=child.page_end,
                    section_path=section_path,
                    order_index=parent_index * 1000 + child_index,
                )
            )

    return ChunkingResult(parents=parents, children=rebuilt_children)


def _coalesce_short(children: list[Chunk], min_size: int) -> list[Chunk]:
    """薄包装, 委托给共享实现(见 base.coalesce_short_chunks)."""
    return coalesce_short_chunks(children, min_size)


__all__ = ["FlatText", "chunk_fixed", "chunk_recursive", "split_sentences"]
