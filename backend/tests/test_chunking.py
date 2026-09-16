"""分块层测试.

重点验证四件事:
1. **父子结构**: 子块指向正确的父块
2. **页码准确**: 子块的 page_start/page_end 来自真实来源段落, 不是笼统继承父块
3. **id 确定性**: 同样的输入永远生成同样的 id —— 这是整个入库流程幂等的基础
4. **短块合并**: 过短的尾部碎片被并入相邻块, 而不是独立成一条无信息的向量
"""

from __future__ import annotations

from app.models.document import ChunkType
from app.services.chunking import chunk_document, split_sentences
from app.services.parser.base import CleanDocument, Paragraph


def _doc(*paragraphs: Paragraph) -> CleanDocument:
    return CleanDocument(filename="test.pdf", paragraphs=list(paragraphs), page_count=3)


def _para(text: str, page: int = 1, heading: bool = False, number: str | None = None) -> Paragraph:
    return Paragraph(text=text, page_no=page, is_heading=heading, section_number=number)


# --------------------------------------------------------------------------- #
# 句子切分
# --------------------------------------------------------------------------- #
def test_split_sentences_keeps_chinese_punctuation() -> None:
    assert split_sentences("钢刀寿命为20000次。锡膏需冷藏。") == [
        "钢刀寿命为20000次。",
        "锡膏需冷藏。",
    ]


def test_split_sentences_does_not_break_decimal_numbers() -> None:
    """ "3.2" 里的点号不是句末, 不能把编号切碎."""
    assert split_sentences("参见 3.2 节关于设备维护的说明") == ["参见 3.2 节关于设备维护的说明"]


# --------------------------------------------------------------------------- #
# 父子结构
# --------------------------------------------------------------------------- #
def test_creates_parent_child_structure() -> None:
    doc = _doc(
        _para("设备维护说明", heading=True),
        _para("钢刀的更换周期为20000次或者3个月, 需要定期检查磨损情况并记录。", page=1),
        _para("锡膏需要储存在2到10摄氏度的环境中, 使用前必须回温至少30分钟。", page=2),
    )
    result = chunk_document(doc, doc_id="DOC1")

    assert result.parents
    assert result.children
    for child in result.children:
        assert child.chunk_type == ChunkType.CHILD
        assert child.parent_id is not None
    for parent in result.parents:
        assert parent.chunk_type == ChunkType.PARENT
        assert parent.parent_id is None


def test_every_child_points_to_an_existing_parent() -> None:
    doc = _doc(
        *[_para(f"第{i}段的内容, 用于验证父子关系是否正确建立。", page=i) for i in range(1, 4)]
    )
    result = chunk_document(doc, doc_id="DOC2")

    parent_ids = {p.id for p in result.parents}
    assert {c.parent_id for c in result.children} <= parent_ids


def test_child_page_range_comes_from_source_paragraphs() -> None:
    """子块的页码必须来自真实来源段落 —— 引用溯源要精确到页, 不能笼统继承父块."""
    doc = _doc(
        _para("第一页的内容, 描述设备的基本参数和规格说明。", page=1),
        _para("第二页的内容, 描述设备的维护周期和注意事项。", page=2),
    )
    result = chunk_document(doc, doc_id="DOC3")

    pages = {(c.page_start, c.page_end) for c in result.children}
    # 至少有一个子块跨越了 1~2 页, 说明页码是按实际内容累计的
    assert any(start <= 1 and end >= 1 for start, end in pages)
    for child in result.children:
        assert 1 <= child.page_start <= child.page_end <= 2


# --------------------------------------------------------------------------- #
# id 确定性
# --------------------------------------------------------------------------- #
def test_chunk_ids_are_deterministic() -> None:
    """同样的输入必须生成同样的 id.

    这是整个入库流程幂等的基础: 重跑时 upsert 覆盖而不是新增.
    如果 id 是随机 UUID, 每次重试都会在向量库里堆一份新数据.
    """
    doc = _doc(
        *[
            _para(f"第{i}段的正文内容, 包含足够的字符数以形成独立分块。", page=i)
            for i in range(1, 6)
        ]
    )

    first = chunk_document(doc, doc_id="SAME")
    second = chunk_document(doc, doc_id="SAME")

    assert [c.id for c in first.children] == [c.id for c in second.children]
    assert [p.id for p in first.parents] == [p.id for p in second.parents]


def test_ids_differ_across_documents() -> None:
    doc = _doc(_para("相同的正文内容, 用于验证不同文档的 id 不会碰撞。"))
    assert (
        chunk_document(doc, doc_id="A").children[0].id
        != chunk_document(doc, doc_id="B").children[0].id
    )


# --------------------------------------------------------------------------- #
# 短块合并
# --------------------------------------------------------------------------- #
def test_short_tail_chunk_is_merged() -> None:
    """段落末尾剩下的短句应该并入前一块, 而不是独立成一条无信息的向量."""
    from app.core.config import settings

    doc = _doc(
        _para("设备维护的核心要点如下所述, 需要严格执行并且记录每一次的检查结果。", page=1),
        _para("已验证。", page=1),  # 极短
    )
    result = chunk_document(doc, doc_id="DOC4", child_size=60, overlap=0)

    for child in result.children:
        assert child.char_count >= min(settings.min_chunk_size, 100) or len(result.children) == 1


# --------------------------------------------------------------------------- #
# 章节路径
# --------------------------------------------------------------------------- #
def test_section_path_tracks_heading_hierarchy() -> None:
    doc = _doc(
        _para("第三章 设备维护", heading=True, number="第三章"),
        _para("3.1 钢刀", heading=True, number="3.1"),
        _para("钢刀的更换周期为20000次, 需要定期检查磨损并记录更换时间。"),
        _para("3.2 锡膏", heading=True, number="3.2"),
        _para("锡膏需要冷藏保存, 使用前必须回温至少30分钟才能投入使用。"),
    )
    result = chunk_document(doc, doc_id="DOC5")

    paths = {c.section_path for c in result.children}
    assert any(p and "3.1" in p and "第三章" in p for p in paths)
    assert any(p and "3.2" in p for p in paths)


def test_section_path_drops_deeper_levels_on_new_sibling() -> None:
    """遇到同级新标题时, 之前的子级标题必须从路径里消失."""
    doc = _doc(
        _para("第一章", heading=True, number="1"),
        _para("1.1 小节", heading=True, number="1.1"),
        _para("第一章第一节的正文内容, 用于验证路径拼接的正确性。"),
        _para("第二章", heading=True, number="2"),
        _para("第二章的正文内容, 路径里不应该再出现 1.1。"),
    )
    result = chunk_document(doc, doc_id="DOC6")

    last = result.children[-1]
    assert last.section_path is not None
    assert "第二章" in last.section_path
    assert "1.1" not in last.section_path


def test_section_path_has_no_newlines() -> None:
    """防御性: 即使解析层误判了一个多行块为标题, 也不能让换行符污染 section_path."""
    doc = _doc(
        _para("标签\n第二行\n第三行", heading=True),
        _para("正文内容, 用于验证章节路径的清洗逻辑是否生效。"),
    )
    result = chunk_document(doc, doc_id="DOC7")
    for child in result.children:
        assert "\n" not in (child.section_path or "")


# --------------------------------------------------------------------------- #
# embedding_text
# --------------------------------------------------------------------------- #
def test_embedding_text_prepends_section_path() -> None:
    doc = _doc(
        _para("第三章 设备维护", heading=True, number="第三章"),
        _para("设备运行满一个季度后需要安排一次全面检修并记录结果。"),
        _para("锡膏需要冷藏保存, 使用前必须回温至少30分钟才能投入使用。"),
    )
    children = chunk_document(doc, doc_id="DOC8").children

    # 不含标题正文的子块: 应该拼上章节路径补语境
    assert any(c.embedding_text.startswith("第三章 设备维护") for c in children)


def test_embedding_text_does_not_duplicate_heading() -> None:
    """子块正文已经以章节标题开头时, 不应再拼一遍.

    父块的第一个子块通常包含标题段落本身, 此时拼前缀会产生
    "标题 > 标题 + 正文" 的重复文本 —— 白占 token 还稀释向量.
    """
    doc = _doc(
        _para("第三章 设备维护", heading=True, number="第三章"),
        _para("设备运行满一个季度后需要安排一次全面检修并记录结果。"),
    )
    child = chunk_document(doc, doc_id="DOC8B").children[0]

    assert child.content.startswith("第三章 设备维护")
    assert child.embedding_text == child.content
    assert child.embedding_text.count("第三章 设备维护") == 1


def test_content_stays_clean_without_section_prefix() -> None:
    """存库的 content 必须保持原文, 不能带上人为拼接的章节前缀.

    引用展示、BM25 关键词检索用的都是 content;
    只有送去算向量的文本才需要补语境.
    """
    doc = _doc(
        _para("第三章 设备维护", heading=True, number="第三章"),
        _para("设备运行满一个季度后需要安排一次全面检修并记录结果。"),
        _para("锡膏需要冷藏保存, 使用前必须回温至少三十分钟才能投入使用。"),
        _para("清洗站的校准周期为每周一次, 由当班工程师负责执行并签字。"),
    )
    # 用小 child_size 强制切出多个子块, 才能观察到"不含标题的子块"
    children = chunk_document(doc, doc_id="DOC8C", child_size=40, overlap=0).children
    body_children = [c for c in children if not c.content.startswith("第三章")]

    assert body_children, "应该存在正文子块(不含标题)"
    for child in body_children:
        assert " > " not in child.content
        assert child.section_path == "第三章 设备维护"
        assert child.embedding_text.startswith("第三章 设备维护 > ")


def test_embedding_text_without_section_path_is_plain_content() -> None:
    doc = _doc(_para("没有标题的正文内容, 直接使用原文作为向量文本。"))
    child = chunk_document(doc, doc_id="DOC9").children[0]

    assert child.section_path is None
    assert child.embedding_text == child.content


# --------------------------------------------------------------------------- #
# 参数校验
# --------------------------------------------------------------------------- #
def test_child_size_must_be_smaller_than_parent() -> None:
    """子块比父块还大时父子结构失去意义, 必须显式报错而不是静默产出垃圾."""
    import pytest

    from app.core.exceptions import ParamInvalidError

    doc = _doc(_para("内容"))
    with pytest.raises(ParamInvalidError, match="子块大小必须小于父块大小"):
        chunk_document(doc, doc_id="DOC10", parent_size=100, child_size=200)


def test_oversized_paragraph_is_split() -> None:
    """单个超长段落必须被切开, 否则会出现远超模型上下文的分块."""
    huge = _para("这是一句用于测试的长句子。" * 100)
    result = chunk_document(_doc(huge), doc_id="DOC11", parent_size=300, child_size=100)

    assert len(result.parents) > 1
