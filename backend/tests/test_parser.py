"""解析层测试.

分成两组, 边界清晰:
- **纯函数测试**(清洗、标题识别): 用内存数据, 不碰文件系统
- **PDF 端到端测试**: 用 conftest 生成的合成 PDF, 验证坐标/分栏/页眉页脚处理
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.parser import clean_block_text, clean_document, parse_pdf
from app.services.parser.base import ParsedPage, TextBlock
from app.services.parser.cleaner import is_cjk, merge_broken_lines, normalize_whitespace
from app.services.parser.pdf_parser import (
    _body_font_size,
    _detect_running_headers,
    _mark_heading,
    extract_section_number,
)


# --------------------------------------------------------------------------- #
# 文本清洗
# --------------------------------------------------------------------------- #
def test_normalize_removes_invisible_characters() -> None:
    """零宽字符会污染 embedding, 必须清掉."""
    text = "钢刀\u200b寿命\ufeff为\u00ad20000次"
    assert normalize_whitespace(text) == "钢刀寿命为20000次"


def test_normalize_removes_private_use_area_glyphs() -> None:
    """图标字体的私用区码点(\ue621 这类)必须清掉.

    真实 PDF 里很常见: 简历/PPT 导出的 PDF 会把图标渲染成 iconfont,
    在文本层留下无意义的私用区字符. 它们会混进 embedding 稀释语义,
    并在引用展示时渲染成乱码方块.
    """
    text = "年\ue621龄 22 岁\ue622性\ue61f别 男"
    cleaned = normalize_whitespace(text)

    assert "\ue621" not in cleaned
    assert "\ue622" not in cleaned
    assert all(ch in cleaned for ch in "年龄性男")


def test_normalize_collapses_repeated_spaces() -> None:
    assert normalize_whitespace("钢刀    寿命  \t 20000") == "钢刀 寿命 20000"


def test_mid_sentence_line_break_is_merged() -> None:
    """PDF 的换行是排版换行, 不是语义换行 —— 必须合并."""
    text = "钢刀的更换周期为 20000 次或 3 个\n月"
    assert merge_broken_lines(text) == "钢刀的更换周期为 20000 次或 3 个月"


def test_sentence_end_prevents_merge() -> None:
    """上一行已以句号结束, 语义完整, 不应该被合并."""
    text = "第一句话已经结束。\n第二句话另起一行。"
    assert merge_broken_lines(text) == text


def test_bullet_start_prevents_merge() -> None:
    """下一行是新条目时不能合并, 否则会把列表项粘成一段."""
    text = "本设备包含以下部件\n（1）钢刀\n（2）锡膏"
    assert merge_broken_lines(text).count("\n") == 2


def test_pure_page_number_is_dropped() -> None:
    assert clean_block_text("12") == ""
    assert clean_block_text("- 7 -") == ""
    # 但含页码的完整句子不应被误删
    assert clean_block_text("第 12 页共 30 页") != ""


def test_is_cjk_recognises_chinese_and_punctuation() -> None:
    assert is_cjk("钢")
    assert is_cjk("，")
    assert not is_cjk("A")
    assert not is_cjk("1")


def test_merge_uses_no_space_between_cjk() -> None:
    """中文之间不应插入空格, 否则会切出无意义的词边界."""
    text = "设备维护的核心要点是定期检查钢刀\n磨损情况并记录更换时间"
    merged = merge_broken_lines(text)
    assert "\n" not in merged
    assert "钢刀磨损" in merged


def test_merge_inserts_space_between_latin_words() -> None:
    """英文断行合并要补空格, 否则两个单词会粘成一个不存在的词."""
    merged = merge_broken_lines("the maintenance interval for each\nstation in the line")
    assert "each station" in merged


def test_short_label_line_is_not_merged() -> None:
    """很短的标签行(如 "关键词:")不参与合并.

    这是刻意的保守策略: 短行既可能是被切断的句子开头, 也可能是独立标签.
    宁可漏合并(表现为分块里多一个换行), 也不要把标签和正文粘成一句
    —— 后者会污染语义, 更难发现.
    """
    assert merge_broken_lines("关键词:\n检索 向量 分块") == "关键词:\n检索 向量 分块"


# --------------------------------------------------------------------------- #
# 标题识别
# --------------------------------------------------------------------------- #
def _block(text: str, font_size: float) -> TextBlock:
    return TextBlock(text=text, page_no=1, x0=0, y0=0, x1=100, y1=10, font_size=font_size)


def test_body_font_size_uses_mode_not_mean() -> None:
    """正文字号必须取众数 —— 均值会被少量大标题拉高, 导致真标题"不够大"."""
    pages = [
        ParsedPage(
            page_no=1,
            width=595,
            height=842,
            blocks=[
                _block("正文段落" * 40, 10.0),
                _block("正文段落" * 30, 10.0),
                _block("大标题", 20.0),
            ],
        )
    ]
    assert _body_font_size(pages) == 10.0


def test_slightly_larger_single_line_is_heading() -> None:
    """真实简历里标题只比正文大 1 磅, 绝对差值规则必须能命中."""
    assert _mark_heading(_block("教育背景", 10.4), 9.4).is_heading is True


def test_slightly_larger_but_multiline_is_not_heading() -> None:
    """表格/表单标签块字号偏大但是多行, 必须排除.

    这是本项目真实踩到的坑: 简历里 "年　　龄\\n22 岁\\n性　　别\\n男"
    被 PyMuPDF 合并成一个 13pt 的块, 一度被误判为标题.
    """
    text = "年　　龄\n22 岁\n性　　别\n男"
    assert _mark_heading(_block(text, 13.1), 9.4).is_heading is False


def test_body_text_is_not_heading() -> None:
    long_sentence = "这是一段很长的正文内容, 用来确认它不会被误判成标题。"
    assert _mark_heading(_block(long_sentence, 10.0), 9.4).is_heading is False


def test_numbered_line_is_heading_even_at_body_size() -> None:
    """结构化文档用编号标记层级, 即使字号与正文相同也应识别."""
    assert _mark_heading(_block("3.2 设备维护", 10.0), 10.0).is_heading is True
    assert _mark_heading(_block("第三章 设备维护", 10.0), 10.0).is_heading is True


def test_model_number_is_not_mistaken_for_section_number() -> None:
    """ "1333 机种" 以数字开头, 但不能被当成章节编号."""
    assert extract_section_number("1333 机种钢刀更换规范") is None
    assert extract_section_number("3.2 钢刀更换规范") == "3.2"


def test_overlong_text_is_never_heading() -> None:
    assert _mark_heading(_block("长" * 100, 30.0), 10.0).is_heading is False


# --------------------------------------------------------------------------- #
# 页眉页脚检测
# --------------------------------------------------------------------------- #
def _page(page_no: int, first: str, middle: str, last: str) -> ParsedPage:
    return ParsedPage(
        page_no=page_no,
        width=595,
        height=842,
        blocks=[
            TextBlock(text=first, page_no=page_no, x0=0, y0=10, x1=100, y1=20, font_size=8),
            TextBlock(text=middle, page_no=page_no, x0=0, y0=100, x1=500, y1=120, font_size=10),
            TextBlock(text=last, page_no=page_no, x0=0, y0=800, x1=100, y1=810, font_size=8),
        ],
    )


def test_repeated_header_and_footer_are_detected() -> None:
    pages = [
        _page(i, "设备维护手册", f"第 {i} 页的正文内容各不相同", f"Page {i} of 5")
        for i in range(1, 6)
    ]
    detected = _detect_running_headers(pages)

    assert "设备维护手册" in detected
    # 页码每页不同, 但数字归一化后一致 —— 这正是"归一化"设计的意义
    assert "page # of #" in detected


def test_unique_content_is_not_detected() -> None:
    """每页内容都不同的文档, 不应该误删任何东西.

    注意测试数据必须是**真正不同**的文本. 如果只让数字变化,
    归一化之后它们会变成同一个 key, 那就确实符合"重复行"的定义了 ——
    这是设计使然, 不是 bug.
    """
    firsts = ["产品规格说明", "安装步骤指引", "故障排查手册", "维护保养计划", "附录与术语"]
    middles = [
        "本页介绍设备的额定电压与功率参数",
        "本页说明安装前的环境检查清单",
        "本页列举常见报警代码与处理方法",
        "本页给出季度保养的时间安排",
        "本页汇总全文使用的专有名词",
    ]
    lasts = [
        "文档编号 A-100",
        "文档编号 B-200",
        "文档编号 C-300",
        "文档编号 D-400",
        "文档编号 E-500",
    ]

    pages = [_page(i, firsts[i - 1], middles[i - 1], lasts[i - 1]) for i in range(1, 6)]
    assert _detect_running_headers(pages) == set()


def test_short_document_skips_header_detection() -> None:
    """少于 3 页时样本太少, 误杀风险高, 应直接跳过检测."""
    pages = [_page(1, "同一行", "内容一", "尾一"), _page(2, "同一行", "内容二", "尾二")]
    assert _detect_running_headers(pages) == set()


# --------------------------------------------------------------------------- #
# PDF 端到端
# --------------------------------------------------------------------------- #
def test_parse_extracts_pages_and_text(sample_pdf: Path) -> None:
    parsed = parse_pdf(sample_pdf)

    assert parsed.page_count == 3
    assert parsed.char_count > 200
    assert parsed.is_scanned is False
    assert parsed.filename == "sample.pdf"


def test_running_header_removed_from_output(sample_pdf: Path) -> None:
    """页眉在每一页都出现, 解析结果里不应该再看到它."""
    parsed = parse_pdf(sample_pdf)
    all_text = "\n".join(b.text for b in parsed.all_blocks())
    assert "Equipment Maintenance Guide" not in all_text


def test_page_number_footer_removed_from_output(sample_pdf: Path) -> None:
    parsed = parse_pdf(sample_pdf)
    all_text = "\n".join(b.text for b in parsed.all_blocks())
    assert "Page 1 of 3" not in all_text


def test_headings_detected_by_font_size(sample_pdf: Path) -> None:
    """18pt 主标题与 12pt 章节标题都应该被识别出来."""
    parsed = parse_pdf(sample_pdf)
    headings = {b.text for b in parsed.all_blocks() if b.is_heading}

    assert "DocMind Test Document" in headings
    assert any("Section One" in h for h in headings)
    # 10pt 的正文不应该被误判
    assert not any("This document states" in h for h in headings)


def test_blocks_carry_page_numbers(sample_pdf: Path) -> None:
    """页码是引用溯源的基础, 每个块都必须带."""
    parsed = parse_pdf(sample_pdf)
    for page in parsed.pages:
        for block in page.blocks:
            assert block.page_no == page.page_no


def test_clean_document_produces_numbered_paragraphs(sample_pdf: Path) -> None:
    parsed = parse_pdf(sample_pdf)
    cleaned = clean_document(parsed)

    assert cleaned.paragraphs
    assert cleaned.page_count == 3
    assert all(p.page_no >= 1 for p in cleaned.paragraphs)
    assert cleaned.char_count > 0


def test_parse_rejects_corrupt_file(tmp_path: Path) -> None:
    """损坏的文件必须给出明确错误, 而不是抛裸异常或静默返回空."""
    from app.core.exceptions import DocumentParseError

    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"this is definitely not a pdf")

    with pytest.raises(DocumentParseError):
        parse_pdf(bad)
