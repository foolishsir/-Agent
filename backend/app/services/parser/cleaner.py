"""文本清洗.

PDF 抽取出来的文本是"排版残留", 主要噪音有四类:

1. **断行**: PDF 的换行是排版换行, 不是语义换行.
   "钢刀的更换周期为 20000 次或 3 个\n月" 需要在清洗阶段合并回一句.
2. **零宽字符与软连字符**: 从网页/LaTeX 生成的 PDF 里很常见, 会污染 embedding.
3. **页码/装饰行**: 单独一行只有一个数字, 属于噪音.
4. **页内重复行**: 某些模板会在同一页重复表头.

清洗的目标不是"变干净好看", 而是**提升 embedding 质量** ——
噪音字符会让语义向量偏移, 直接拉低检索命中率.
"""

from __future__ import annotations

import re
import unicodedata

from app.core.logging import get_logger, log_kv
from app.services.parser.base import CleanDocument, Paragraph, ParsedDocument
from app.services.parser.pdf_parser import extract_section_number

logger = get_logger("docmind.parser.cleaner")

# --------------------------------------------------------------------------- #
# 正则
# --------------------------------------------------------------------------- #
#: 零宽字符 + 软连字符 + BOM. 用 \u 转义书写, 避免源码里出现不可见字符
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff\u00ad]")

#: Unicode 私用区(Private Use Area).
#: PDF 里的私用区字符几乎总是**图标字体**(iconfont)的字形编码 —— 它们
#: 在文本层表现为 \ue621 这种无意义的码点, 对语义毫无贡献, 却会:
#:   1. 混进 embedding 文本, 稀释有效语义
#:   2. 出现在引用展示里变成乱码方块
#: 代价: 极少数把生僻汉字映射进私用区的字体, 其字符也会被一并删除.
#: 权衡下来, 去图标噪音的收益远大于偶发丢一个生僻字.
#: 实测某份简历里每页都有 5 个这样的图标码点.
_PUA_RE = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd]")

#: 连续空白(不含换行, 换行要保留给断行合并逻辑判断)
_INLINE_SPACE_RE = re.compile(r"[ \t\u00a0\u3000]{2,}")

#: 单独成行的页码/装饰数字
_PAGE_NUMBER_RE = re.compile(r"^\s*[-—–\[\(]*\s*\d{1,4}\s*[-—–\]\)]*\s*$")

#: 句末标点(中英文) —— 行尾是这些字符时, 说明语义已完整, 不应与下一行合并
_SENTENCE_END = "。！？；：.!?;:…\"'）)】》」』"

#: 行首是这些字符时, 说明是新条目的开始, 不应与上一行合并
_LINE_START_MARKERS = re.compile(
    r"^\s*(?:[·•▪◦●○◆◇■□▲△★☆\-–—*]|"
    r"[（(]\s*\d+\s*[)）]|"
    r"\d{1,3}\s*[、.．)）]|"
    r"[一二三四五六七八九十]{1,3}\s*[、.．]|"
    r"第\s*[一二三四五六七八九十百零〇\d]{1,6}\s*[章节条部分篇])"
)

_CJK_RANGES = (
    (0x4E00, 0x9FFF),  # CJK 统一表意文字
    (0x3400, 0x4DBF),  # 扩展 A
    (0x3000, 0x303F),  # CJK 标点
    (0xFF00, 0xFFEF),  # 全角字符
    (0xF900, 0xFAFF),  # 兼容表意文字
)


def is_cjk(char: str) -> bool:
    """判断字符是否是中日韩文字/标点."""
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def _ends_with_cjk(text: str) -> bool:
    return bool(text) and is_cjk(text[-1])


def _starts_with_cjk(text: str) -> bool:
    return bool(text) and is_cjk(text[0])


# --------------------------------------------------------------------------- #
# 单块清洗
# --------------------------------------------------------------------------- #
def normalize_whitespace(text: str) -> str:
    """Unicode 归一化 + 去除不可见字符与图标码点 + 压缩连续空白."""
    # NFKC: 把全角字母数字、兼容字符统一成标准形式.
    # 注意 NFKC 会把全角标点也转换, 对中文正文一般是有益的.
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_RE.sub("", text)
    text = _PUA_RE.sub("", text)
    # 统一各种换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INLINE_SPACE_RE.sub(" ", text)
    # 去掉每行首尾空白, 但保留空行结构
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def merge_broken_lines(text: str) -> str:
    """合并被排版切断的行.

    判断规则: 上一行末尾**没有句末标点**, 且下一行**不是新条目的开头**, 则合并.

    中英文的处理不同:
    - 中文: 直接拼接(中文没有词间空格)
    - 英文: 插入一个空格(否则 "informa" + "tion" 会粘成 "information" 的假象,
      而真实情况是 "distri" + "bution" 这种断词, 加空格反而是错的 —— 但
      加空格的错误率远低于不加空格, 所以选择保守方案)
    """
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) <= 1:
        return text.strip()

    merged: list[str] = [lines[0]]
    for line in lines[1:]:
        previous = merged[-1]
        if _should_merge(previous, line):
            separator = "" if (_ends_with_cjk(previous) or _starts_with_cjk(line)) else " "
            merged[-1] = f"{previous}{separator}{line}"
        else:
            merged.append(line)

    return "\n".join(merged)


def _should_merge(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    # 上一行已经以句末标点结束 → 语义完整, 不合并
    if previous.rstrip().endswith(tuple(_SENTENCE_END)):
        return False
    # 下一行是新条目/新标题的开头 → 不合并
    if _LINE_START_MARKERS.match(current):
        return False
    # 上一行很短(如 "关键词:" 这类标签行) → 保守起见不合并
    return not len(previous.rstrip()) < 6


def clean_block_text(text: str) -> str:
    """对单个文本块做完整清洗."""
    text = normalize_whitespace(text)
    if not text:
        return ""
    text = merge_broken_lines(text)
    # 清理后如果只剩下一个孤立数字, 判定为页码噪音
    if _PAGE_NUMBER_RE.match(text):
        return ""
    return text.strip()


# --------------------------------------------------------------------------- #
# 整份文档清洗
# --------------------------------------------------------------------------- #
def clean_document(parsed: ParsedDocument) -> CleanDocument:
    """把解析结果清洗成干净的段落序列.

    输出是 ``list[Paragraph]``, 每个段落自带页码 —— 这样后续分块时
    "分块 → 页码"的映射是天然成立的, 不需要维护字符偏移量表.
    """
    paragraphs: list[Paragraph] = []
    seen_on_page: set[tuple[int, str]] = set()
    dropped = 0

    for page in parsed.pages:
        for block in page.blocks:
            text = clean_block_text(block.text)
            if not text:
                dropped += 1
                continue

            # 页内去重: 同一页出现完全相同的段落(模板重复的表头/水印)只保留一次
            fingerprint = (page.page_no, text)
            if fingerprint in seen_on_page:
                dropped += 1
                continue
            seen_on_page.add(fingerprint)

            paragraphs.append(
                Paragraph(
                    text=text,
                    page_no=page.page_no,
                    is_heading=block.is_heading,
                    section_number=extract_section_number(text) if block.is_heading else None,
                )
            )
    # 跨页去重: 有些模板会在连续多页重复同一段说明文字
    paragraphs = _dedupe_across_pages(paragraphs)

    result = CleanDocument(
        filename=parsed.filename,
        paragraphs=paragraphs,
        page_count=parsed.page_count,
        metadata=parsed.metadata,
    )

    log_kv(
        logger,
        "cleaner.done",
        file=parsed.filename,
        paragraphs=len(paragraphs),
        chars=result.char_count,
        headings=sum(1 for p in paragraphs if p.is_heading),
        dropped=dropped,
    )
    return result


def _dedupe_across_pages(paragraphs: list[Paragraph]) -> list[Paragraph]:
    """去掉在不同页上完全重复的长段落.

    只对**较长**的段落做跨页去重(长度 >= 40 字).
    短段落如"合计"、"备注"在业务上可能是合法重复, 删掉会丢信息.
    """
    seen: set[str] = set()
    result: list[Paragraph] = []
    for paragraph in paragraphs:
        if paragraph.char_count >= 40:
            key = paragraph.text
            if key in seen:
                continue
            seen.add(key)
        result.append(paragraph)
    return result
