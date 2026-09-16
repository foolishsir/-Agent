"""父子块切分(Small-to-Big).

解决的核心矛盾
--------------
单层分块存在一个无解的取舍:

- 块**小**(如 300 字): 向量语义聚焦, 检索准; 但喂给 LLM 的上下文缺前因后果, 答案不完整
- 块**大**(如 1500 字): 上下文完整; 但向量被稀释, 检索命中率下降

**检索需要小粒度, 生成需要大粒度** —— 这是两个不同层级的诉求, 单层分块不可能同时满足.

方案: 索引时切两层
------------------
::

    父块 (1500 字, 按章节 + 长度切)
     ├── 子块1 (300 字)  ← 只有子块进向量库, 参与检索
     ├── 子块2 (300 字)
     └── 子块3 (300 字)

    检索命中子块2 → 通过 parent_id 回查父块 → 把父块全文喂给 LLM

额外收益: 多个子块命中同一父块时**去重合并**, 只送一份父块.
实际送进 Prompt 的 token 反而比"直接塞 20 个子块"更少且更完整 ——
这是一个少见的"精度和成本同时改善"的方案.

为什么不做语义分块(Embedding 相似度突变点)
-----------------------------------------
语义分块效果提升有限, 但代价是**每个句子都要算一次 embedding**,
索引成本翻数倍. 用章节结构 + 递归切分已经能拿到大部分收益,
剩下的用 Rerank 补更划算.
"""

from __future__ import annotations

import re

from app.core.logging import get_logger, log_kv
from app.models.document import ChunkType
from app.services.chunking.base import (
    Chunk,
    ChunkingResult,
    coalesce_short_chunks,
    make_child_id,
    make_parent_id,
)
from app.services.chunking.params import ChunkParams, split_sentences
from app.services.parser.base import CleanDocument, Paragraph

logger = get_logger("docmind.chunking")

#: section_path 字段长度上限(与 ORM 的 String(512) 对齐)
_MAX_SECTION_PATH = 500


def chunk_parent_child(document: CleanDocument, doc_id: str, params: ChunkParams) -> ChunkingResult:
    """把清洗后的文档切成父子块.

    这是默认策略, 也是效果最好的一种:
    父块按"章节标题 + 长度"切, 子块在父块内按句子边界切并带重叠.
    子块进向量库保检索精度, 父块喂给模型保上下文完整.
    """
    params.validate()

    parents: list[Chunk] = []
    children: list[Chunk] = []

    parent_index = 0
    for section_path, paragraphs in _iter_sections(document):
        for group in _group_paragraphs(paragraphs, params.parent_size):
            parent_id = make_parent_id(doc_id, parent_index)
            parent_text = "\n\n".join(p.text for p in group)

            parents.append(
                Chunk(
                    id=parent_id,
                    doc_id=doc_id,
                    parent_id=None,
                    chunk_type=ChunkType.PARENT,
                    content=parent_text,
                    page_start=min(p.page_no for p in group),
                    page_end=max(p.page_no for p in group),
                    section_path=section_path,
                    order_index=parent_index,
                )
            )

            children.extend(
                _split_parent_into_children(
                    group,
                    doc_id,
                    parent_index,
                    parent_id,
                    section_path,
                    params,
                )
            )

            parent_index += 1

    result = ChunkingResult(parents=parents, children=children)
    log_kv(
        logger,
        "chunking.done",
        strategy=params.strategy,
        file=document.filename,
        parents=len(parents),
        children=len(children),
        avg_child_chars=round(sum(c.char_count for c in children) / len(children), 1)
        if children
        else 0,
    )
    return result


# --------------------------------------------------------------------------- #
# 章节拆分
# --------------------------------------------------------------------------- #
def _iter_sections(document: CleanDocument) -> list[tuple[str | None, list[Paragraph]]]:
    """按标题把段落分组成章节, 并生成层级化的 section_path.

    层级推断规则: 标题编号里的点号个数决定层级("3" → 1 级, "3.2" → 2 级).
    没有编号的标题一律当作 1 级.

    示例::

        第三章 设备维护          →  "第三章 设备维护"
        3.1 钢刀                 →  "第三章 设备维护 > 3.1 钢刀"
        3.2 锡膏                 →  "第三章 设备维护 > 3.2 锡膏"
        第四章 ...               →  "第四章 ..."          (第三章及其子级被替换)
    """
    sections: list[tuple[str | None, list[Paragraph]]] = []
    heading_stack: dict[int, str] = {}
    current: list[Paragraph] = []
    current_path: str | None = None

    def flush() -> None:
        if current:
            sections.append((current_path, list(current)))
            current.clear()

    for paragraph in document.paragraphs:
        if paragraph.is_heading:
            flush()
            level = _heading_level(paragraph.section_number)
            # 同级或更深的旧标题全部失效
            for existing in [lv for lv in heading_stack if lv >= level]:
                del heading_stack[existing]
            heading_stack[level] = paragraph.text
            current_path = _build_section_path(heading_stack)

        current.append(paragraph)

    flush()
    return sections


def _heading_level(section_number: str | None) -> int:
    """由编号推断标题层级."""
    if not section_number:
        return 1
    if section_number[0].isdigit():
        # "3.2.1" → 3 级
        return min(section_number.count(".") + 1, 4)
    return 1


def _build_section_path(stack: dict[int, str]) -> str:
    """拼出层级路径, 并做一次清洗.

    标题理论上都是单行, 但解析层可能把带换行的块误判为标题(不同文档的字体差异很大).
    在这里做归一化是**防御性**的: 即使上游判断失误, 也不会让换行符污染
    后续拼给 embedding 的文本.
    """
    parts = [re.sub(r"\s+", " ", stack[level]).strip() for level in sorted(stack)]
    return " > ".join(part for part in parts if part)[:_MAX_SECTION_PATH]


# --------------------------------------------------------------------------- #
# 父块分组
# --------------------------------------------------------------------------- #
def _group_paragraphs(paragraphs: list[Paragraph], max_size: int) -> list[list[Paragraph]]:
    """把段落累积成不超过 ``max_size`` 的父块.

    单个段落本身就超过上限时(如一个超长的表格文本), 按句子强行切开 ——
    否则会出现一个几万字、远超模型上下文的分块.
    """
    groups: list[list[Paragraph]] = []
    current: list[Paragraph] = []
    current_len = 0

    for paragraph in paragraphs:
        # 超长单段: 先冲掉已累积的内容, 再把这个段拆开
        if paragraph.char_count > max_size:
            if current:
                groups.append(current)
                current, current_len = [], 0
            groups.extend(_split_oversized_paragraph(paragraph, max_size))
            continue

        if current and current_len + paragraph.char_count > max_size:
            groups.append(current)
            current, current_len = [], 0

        current.append(paragraph)
        current_len += paragraph.char_count

    if current:
        groups.append(current)
    return groups


def _split_oversized_paragraph(paragraph: Paragraph, max_size: int) -> list[list[Paragraph]]:
    """把超长段落按句子拆成多个伪段落."""
    sentences = split_sentences(paragraph.text)
    if not sentences:
        return [[paragraph]]

    groups: list[list[Paragraph]] = []
    bucket: list[str] = []
    bucket_len = 0

    for sentence in sentences:
        if bucket and bucket_len + len(sentence) > max_size:
            groups.append([_rebuild(paragraph, bucket)])
            bucket, bucket_len = [], 0
        bucket.append(sentence)
        bucket_len += len(sentence)

    if bucket:
        groups.append([_rebuild(paragraph, bucket)])
    return groups


def _rebuild(source: Paragraph, sentences: list[str]) -> Paragraph:
    """用一组句子重建段落(保留原页码与层级信息)."""
    return Paragraph(
        text="".join(sentences),
        page_no=source.page_no,
        is_heading=False,
        section_number=None,
    )


# --------------------------------------------------------------------------- #
# 子块切分
# --------------------------------------------------------------------------- #
def _split_parent_into_children(
    group: list[Paragraph],
    doc_id: str,
    parent_index: int,
    parent_id: str,
    section_path: str | None,
    params: ChunkParams,
) -> list[Chunk]:
    """把父块切成带重叠的子块.

    **带句子和页码的单元列表**是这里的关键: 每个句子都记住自己来自哪一页,
    这样子块的 ``page_start/page_end`` 才是准确的, 而不是笼统地继承父块范围.
    引用溯源要精确到页, 这一步不能糊弄.
    """
    child_size = params.child_size
    overlap = params.overlap

    units: list[tuple[str, int]] = []
    for paragraph in group:
        sentences = split_sentences(paragraph.text)
        if not sentences:
            continue
        for sentence in sentences:
            units.append((sentence, paragraph.page_no))

    if not units:
        return []

    chunks: list[Chunk] = []
    bucket: list[tuple[str, int]] = []
    bucket_len = 0

    def flush() -> None:
        nonlocal bucket, bucket_len
        if not bucket:
            return
        chunks.append(
            Chunk(
                id=make_child_id(doc_id, parent_index, len(chunks)),
                doc_id=doc_id,
                parent_id=parent_id,
                chunk_type=ChunkType.CHILD,
                content="".join(text for text, _ in bucket),
                page_start=min(page for _, page in bucket),
                page_end=max(page for _, page in bucket),
                section_path=section_path,
                order_index=parent_index * 1000 + len(chunks),
            )
        )
        bucket = _carry_overlap(bucket, overlap, max_carry=child_size // 2)
        bucket_len = sum(len(text) for text, _ in bucket)

    for text, page_no in units:
        if bucket and bucket_len + len(text) > child_size:
            flush()
        bucket.append((text, page_no))
        bucket_len += len(text)

    if bucket:
        # 最后一个桶可能整份都是重叠内容(和上一个块完全一样), 这种情况直接丢弃
        candidate = "".join(text for text, _ in bucket)
        if not chunks or candidate.strip() != chunks[-1].content.strip():
            chunks.append(
                Chunk(
                    id=make_child_id(doc_id, parent_index, len(chunks)),
                    doc_id=doc_id,
                    parent_id=parent_id,
                    chunk_type=ChunkType.CHILD,
                    content=candidate,
                    page_start=min(page for _, page in bucket),
                    page_end=max(page for _, page in bucket),
                    section_path=section_path,
                    order_index=parent_index * 1000 + len(chunks),
                )
            )

    return coalesce_short_chunks(chunks, params.min_size)


def _carry_overlap(
    bucket: list[tuple[str, int]], overlap: int, max_carry: int | None = None
) -> list[tuple[str, int]]:
    """从已满的桶尾部取出若干**完整句子**作为下一个块的开头.

    为什么按句子而不是按字符取: 按字符取会把一句话切成两半,
    造成"上一块结尾半句、下一块开头半句"的双重噪音.

    ``max_carry`` 是防御性上限: 当 ``overlap`` 配置得接近 ``child_size`` 时,
    不加限制会让每个新块由"几乎全是重叠内容 + 一句话"组成,
    块与块高度冗余, 检索结果中出现大量近似重复.
    """
    if overlap <= 0:
        return []

    limit = overlap if max_carry is None else min(overlap, max_carry)
    carried: list[tuple[str, int]] = []
    total = 0
    for text, page_no in reversed(bucket):
        if total + len(text) > limit and carried:
            break
        carried.append((text, page_no))
        total += len(text)
    carried.reverse()
    return carried
