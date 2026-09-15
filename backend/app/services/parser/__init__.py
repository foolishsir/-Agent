"""文档解析层.

对外只暴露两个函数: ``parse_pdf`` 和 ``clean_document``.
上层(ingest 编排)只依赖这两个入口, 不关心内部的坐标处理细节.
"""

from app.services.parser.base import (
    CleanDocument,
    Paragraph,
    ParsedDocument,
    ParsedPage,
    TextBlock,
)
from app.services.parser.cleaner import clean_block_text, clean_document, normalize_whitespace
from app.services.parser.pdf_parser import parse_pdf

__all__ = [
    "CleanDocument",
    "Paragraph",
    "ParsedDocument",
    "ParsedPage",
    "TextBlock",
    "clean_block_text",
    "clean_document",
    "normalize_whitespace",
    "parse_pdf",
]
