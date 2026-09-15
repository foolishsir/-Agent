"""解析层的数据结构.

为什么要在解析阶段就保留坐标和页码
--------------------------------
很多人写 RAG 直接从 PDF 抽出纯文本字符串就完事, 后面所有信息都丢失了:

- 没有**页码** → 无法做引用溯源("这段话出自第 12 页")
- 没有**坐标** → 无法还原双栏排版, 无法识别页眉页脚
- 没有**字号** → 无法区分标题和正文, 分块只能瞎切

所以这里的 TextBlock 把这三样都保留下来. 这是后续所有环节能做好前提,
也是"能不能识别双栏、能不能去页眉"的关键分水岭.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TextBlock:
    """PDF 页面上的一个文本块(通常是一个段落, 有时是一整块版式文本).

    ``frozen=True`` 是为了能安全地放进 set / dict 做去重统计;
    ``slots=True`` 省内存 —— 一份 200 页文档可能有上万个小对象.
    """

    text: str
    page_no: int  # 从 1 开始, 面向用户展示
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    is_heading: bool = False

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class ParsedPage:
    """一页的解析结果."""

    page_no: int
    width: float
    height: float
    blocks: list[TextBlock] = field(default_factory=list)

    @property
    def text_length(self) -> int:
        return sum(len(b.text) for b in self.blocks)


@dataclass
class ParsedDocument:
    """整份文档的解析结果(尚未清洗)."""

    filename: str
    pages: list[ParsedPage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # 是否疑似扫描件(文本层几乎为空). 上层据此给出明确提示而不是静默返回空结果.
    is_scanned: bool = False

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(p.text_length for p in self.pages)

    def all_blocks(self) -> list[TextBlock]:
        return [block for page in self.pages for block in page.blocks]


@dataclass(frozen=True, slots=True)
class Paragraph:
    """清洗后的一个自然段.

    这是分块层的输入单位. 保留 ``page_no`` 让"分块 → 页码"的映射自然成立,
    不需要维护字符偏移量表.
    """

    text: str
    page_no: int
    is_heading: bool = False
    # 若该段自带编号(如 "3.2"), 这里保留下来, 用于拼接 section_path
    section_number: str | None = None

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class CleanDocument:
    """清洗后的文档."""

    filename: str
    paragraphs: list[Paragraph] = field(default_factory=list)
    page_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return sum(p.char_count for p in self.paragraphs)

    @property
    def text(self) -> str:
        """拼成完整文本(用于统计和调试, 不是分块的输入)."""
        return "\n\n".join(p.text for p in self.paragraphs)
