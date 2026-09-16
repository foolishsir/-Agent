"""分块策略与预览接口的测试.

分块是 RAG 效果的**上限** —— 切坏了后面怎么调 Prompt 都救不回来.
所以这里的测试重点不是"函数能跑", 而是**切出来的东西是否符合预期**:
页码对不对、会不会把句子切断、不同策略是否真的产出不同结构.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.exceptions import ParamInvalidError
from app.services import config_service
from app.services.chunking import (
    AVAILABLE_STRATEGIES,
    ChunkParams,
    chunk_document,
    format_separators,
    parse_separators,
    summarize,
)
from app.services.chunking.strategies import FlatText
from app.services.parser.base import CleanDocument, Paragraph

#: 会被分块相关用例改动的配置字段. 必须**全部**快照还原 ——
#: 少还原一个就会出现"单独跑通过、全量跑失败"的顺序相关故障,
#: 而且现场(后一个用例)与根因(前一个用例)隔得很远, 极难排查.
_CHUNK_SETTINGS = (
    "chunk_strategy",
    "parent_chunk_size",
    "child_chunk_size",
    "chunk_overlap",
    "min_chunk_size",
    "chunk_separators",
    "chunk_keep_heading",
)


@pytest.fixture(autouse=True)
def _restore_chunk_settings() -> Iterator[None]:
    """每个用例前后还原分块配置, 并清掉落盘的运行时配置文件.

    ``settings`` 是进程级单例, 而 ``chunk-apply`` 接口会真的去改它.
    不做隔离的话, 前一个用例留下的 child_size=220 会让后一个用例
    的分块数量断言成片失败 —— 这是本项目真实踩到的测试污染问题.
    """
    snapshot = {name: getattr(settings, name) for name in _CHUNK_SETTINGS}
    path = config_service.runtime_config_path()
    if path.exists():
        path.unlink()

    yield

    for name, value in snapshot.items():
        setattr(settings, name, value)
    if path.exists():
        path.unlink()


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _doc(*paragraphs: Paragraph) -> CleanDocument:
    return CleanDocument(filename="t.pdf", paragraphs=list(paragraphs), page_count=3)


def _para(text: str, page: int = 1, heading: bool = False) -> Paragraph:
    return Paragraph(text=text, page_no=page, is_heading=heading)


def _long_text(pages: int = 3, per_page: int = 6) -> CleanDocument:
    """构造一份足够长的多页文档, 让各种策略都有东西可切."""
    paragraphs: list[Paragraph] = []
    for page in range(1, pages + 1):
        paragraphs.append(_para(f"第{page}章 设备维护", page=page, heading=True))
        for index in range(per_page):
            paragraphs.append(
                _para(
                    f"这是第{page}页的第{index}条说明，内容涉及钢刀更换周期、锡膏储存条件"
                    f"以及清洗站的校准流程，需要定期检查并记录结果。",
                    page=page,
                )
            )
    return _doc(*paragraphs)


# --------------------------------------------------------------------------- #
# 分隔符配置
# --------------------------------------------------------------------------- #
def test_parse_separators_restores_escapes() -> None:
    """界面/.env 里用户没法直接输入换行, 所以要支持 \\n 字面量."""
    seps = parse_separators(r"\n\n|\n|。|，")
    assert seps == ["\n\n", "\n", "。", "，"]


def test_separator_roundtrip() -> None:
    original = ["\n\n", "\n", "。", "！", "，"]
    assert parse_separators(format_separators(original)) == original


def test_separator_delimiter_is_not_comma() -> None:
    """分隔符本身包含 ",，", 所以列表分隔符不能也用逗号 —— 会产生歧义.

    这是选 ``|`` 作为列表分隔符的原因. 用逗号的话,
    "。|，|；" 会被切成 ['。', '，', '；'] 还是 ['。', '，|', '；'] 就说不清了.
    """
    seps = parse_separators("。|，|；")
    assert seps == ["。", "，", "；"]


def test_empty_separators_falls_back_to_default() -> None:
    assert parse_separators("") == parse_separators("|")


# --------------------------------------------------------------------------- #
# 参数校验
# --------------------------------------------------------------------------- #
def test_invalid_strategy_rejected() -> None:
    with pytest.raises(ParamInvalidError, match="不支持的分块策略"):
        ChunkParams(strategy="magic").validate()


def test_overlap_must_be_smaller_than_child() -> None:
    with pytest.raises(ParamInvalidError, match="重叠长度"):
        ChunkParams(child_size=100, overlap=120).validate()


def test_all_strategies_declared_are_usable() -> None:
    """声明的策略必须都能真的跑起来 —— 否则前端下拉框里会出现选不了的选项."""
    doc = _long_text()
    for strategy in AVAILABLE_STRATEGIES:
        result = chunk_document(doc, "DOC", strategy=strategy)
        assert result.children, f"策略 {strategy} 没有产出任何子块"


# --------------------------------------------------------------------------- #
# 固定长度策略(基线)
# --------------------------------------------------------------------------- #
def test_fixed_produces_near_uniform_chunks() -> None:
    result = chunk_document(_long_text(), "DOC", strategy="fixed", child_size=200, overlap=0)
    sizes = [c.char_count for c in result.children]
    # 固定切分的特点就是长度均匀(除最后一块)
    assert max(sizes) - min(sizes) <= 1


def test_fixed_does_not_respect_sentence_boundaries() -> None:
    """这是基线的**特性**, 不是 bug —— 它证明"优化是有意义的".

    固定切分会无视句号把句子切断, 而父子块/递归策略不会.
    对比实验的价值就在于把这种差异量化出来.
    """
    doc = _doc(_para("第一句话已经完整结束了。第二句话也是完整的。第三句话同样完整。"))
    # min_size=0 是刻意的: 默认的"短块合并"会把切出来的尾巴并回前一块,
    # 那样就观察不到"切在句子中间"这个现象了. 这里要测的正是切分点本身.
    result = chunk_document(doc, "DOC", strategy="fixed", child_size=25, overlap=0, min_size=0)

    contents = [c.content for c in result.children]
    assert len(contents) >= 2
    # 至少有一块以非句末标点的字符结尾 → 说明切在了句子中间
    assert any(not c.rstrip().endswith(("。", "！", "？")) for c in contents)


def test_fixed_page_mapping_is_accurate() -> None:
    """页码来自字符级映射, 不能靠猜."""
    result = chunk_document(_long_text(), "DOC", strategy="fixed", child_size=300, overlap=0)
    for child in result.children:
        assert 1 <= child.page_start <= child.page_end <= 3


# --------------------------------------------------------------------------- #
# 递归策略
# --------------------------------------------------------------------------- #
def test_recursive_prefers_paragraph_boundary() -> None:
    """递归切分应尽量在段落边界断开, 而不是硬切字符."""
    doc = _doc(
        _para("第一段的内容，描述设备的基本参数和运行条件。" * 3),
        _para("第二段的内容，描述维护周期和更换标准。" * 3),
        _para("第三段的内容，描述异常处理与记录要求。" * 3),
    )
    result = chunk_document(doc, "DOC", strategy="recursive", child_size=120, overlap=0)

    assert len(result.children) >= 2
    # 每一块都应该以句末标点结尾(说明切在了句子/段落边界)
    assert all(c.content.rstrip().endswith(("。", "！", "？")) for c in result.children)


def test_recursive_handles_text_without_separators() -> None:
    """没有任何分隔符的长文本必须能兜底硬切, 不能死循环或返回空."""
    doc = _doc(_para("无标点长文本" * 200))
    result = chunk_document(doc, "DOC", strategy="recursive", child_size=150, overlap=0)

    assert result.children
    assert all(c.char_count <= 200 for c in result.children)


def test_recursive_overlap_applied_once() -> None:
    """重叠只能应用一次.

    如果每一层递归都加重叠, 嵌套会导致同一段文字被反复计入多个块,
    块与块高度冗余 —— 这是递归切分最容易写错的地方.
    """
    doc = _long_text()
    with_overlap = chunk_document(doc, "DOC", strategy="recursive", child_size=200, overlap=60)
    without = chunk_document(doc, "DOC", strategy="recursive", child_size=200, overlap=0)

    total_with = sum(c.char_count for c in with_overlap.children)
    total_without = sum(c.char_count for c in without.children)

    # 有重叠时字符总量应该略多, 但不应翻倍(翻倍就说明重叠被套娃了)
    assert total_without <= total_with <= total_without * 1.6


# --------------------------------------------------------------------------- #
# 父子块策略
# --------------------------------------------------------------------------- #
def test_parent_child_sets_section_path() -> None:
    result = chunk_document(_long_text(), "DOC", strategy="parent_child")

    assert all(c.section_path for c in result.children)
    assert all("设备维护" in (c.section_path or "") for c in result.children)


def test_parent_child_groups_by_heading() -> None:
    """每个章节标题应该开启一个新的父块分组."""
    result = chunk_document(_long_text(pages=3, per_page=3), "DOC", strategy="parent_child")

    # 3 个章节 → 至少 3 个父块
    assert len(result.parents) >= 3


def test_every_child_belongs_to_a_parent() -> None:
    for strategy in AVAILABLE_STRATEGIES:
        result = chunk_document(_long_text(), "DOC", strategy=strategy)
        parent_ids = {p.id for p in result.parents}
        assert all(c.parent_id in parent_ids for c in result.children), (
            f"{strategy} 有子块没有归属父块"
        )


# --------------------------------------------------------------------------- #
# 通用约束
# --------------------------------------------------------------------------- #
def test_all_strategies_produce_identical_structure() -> None:
    """三种策略必须产出同构的结果 —— 下游检索代码才能不写分支."""
    for strategy in AVAILABLE_STRATEGIES:
        result = chunk_document(_long_text(), "DOC", strategy=strategy)
        assert result.parents and result.children
        assert all(p.chunk_type.value == "parent" for p in result.parents)
        assert all(c.chunk_type.value == "child" for c in result.children)
        assert all(p.parent_id is None for p in result.parents)


def test_chunk_ids_deterministic_across_strategies() -> None:
    """同参数重跑必须产出同样的 id, 否则无法保证入库幂等."""
    first = chunk_document(_long_text(), "DOC", strategy="recursive", child_size=200)
    second = chunk_document(_long_text(), "DOC", strategy="recursive", child_size=200)

    assert [c.id for c in first.children] == [c.id for c in second.children]


def test_min_chunk_size_merges_tiny_chunks() -> None:
    doc = _doc(
        _para("这是一段足够长的正文内容，用来确保它本身不会被判定为过短的块。" * 2),
        _para("短。"),
    )
    result = chunk_document(doc, "DOC", strategy="parent_child", child_size=200, min_size=50)

    assert all(c.char_count >= 3 for c in result.children)


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
def test_summarize_reports_shape() -> None:
    result = chunk_document(_long_text(), "DOC", strategy="parent_child")
    stats = summarize(result, ChunkParams.from_settings())

    assert stats["parents"] == len(result.parents)
    assert stats["children"] == len(result.children)
    assert "avg" in stats["child_chars"]
    assert stats["expansion_ratio"] > 0
    assert 0 <= stats["heading_coverage"] <= 1
    assert "params" in stats


def test_heading_coverage_reflects_strategy() -> None:
    """父子块有标题感知, 覆盖率应显著高于不感知标题的策略.

    这个指标就是"标题识别到底有没有生效"的直接观测点.
    """
    doc = _long_text()
    pc = summarize(chunk_document(doc, "DOC", strategy="parent_child"))
    fx = summarize(chunk_document(doc, "DOC", strategy="fixed", child_size=200))

    assert pc["heading_coverage"] > fx["heading_coverage"]
    assert fx["heading_coverage"] == 0.0


def test_flat_text_page_mapping() -> None:
    flat = FlatText.from_document(_doc(_para("第一页内容", page=1), _para("第二页内容", page=2)))

    assert flat.page_range(0, 5) == (1, 1)
    assert flat.page_range(len(flat) - 3, len(flat)) == (2, 2)
    # 跨页区间
    start, end = flat.page_range(0, len(flat))
    assert (start, end) == (1, 2)


# --------------------------------------------------------------------------- #
# embedding_text 一致性
# --------------------------------------------------------------------------- #
def test_embedding_text_prepends_section_path() -> None:
    from app.services.chunking import build_embedding_text

    text = build_embedding_text("更换周期为 20000 次。", "第三章 设备维护 > 3.2 钢刀")
    assert text.startswith("第三章 设备维护 > 3.2 钢刀")
    assert "更换周期" in text


def test_embedding_text_avoids_duplicate_heading() -> None:
    """正文已经以章节标题开头时不再重复拼接."""
    from app.services.chunking import build_embedding_text

    text = build_embedding_text("教育背景西安石油大学", "教育背景")
    assert text == "教育背景西安石油大学"
    assert text.count("教育背景") == 1


def test_stored_chunks_api_uses_same_embedding_logic(client: TestClient, sample_pdf: Path) -> None:
    """接口返回的 embedding_text 必须与入库时实际使用的一致.

    这是一个真实踩到的 bug: 读取接口用了一版简化拼接(没有去重),
    于是界面上显示成 "章节 > 章节 + 正文", 而实际入库的是去重后的版本.
    用户会基于错误信息判断"送入向量的文本长什么样", 进而做出错误的调参决策.
    """
    doc_id = client.post(
        "/api/v1/documents",
        files={"file": (sample_pdf.name, sample_pdf.read_bytes(), "application/pdf")},
        headers={"X-User-Id": "u-embed-consistency"},
    ).json()["data"]["document"]["id"]

    children = client.get(
        f"/api/v1/documents/{doc_id}/chunks", headers={"X-User-Id": "u-embed-consistency"}
    ).json()["data"]["children"]

    for child in children:
        section = child["section_path"]
        if not section:
            assert child["embedding_text"] == child["content"]
            continue
        leaf = section.split(" > ")[-1]
        # 若正文已以标题开头, 就不应该再拼前缀(否则标题会出现两次)
        if child["content"].lstrip().startswith(leaf):
            assert child["embedding_text"] == child["content"]
        else:
            assert child["embedding_text"].startswith(section + " > ")


# --------------------------------------------------------------------------- #
# 接口
# --------------------------------------------------------------------------- #
def test_chunk_strategies_endpoint(client: TestClient) -> None:
    data = client.get("/api/v1/documents/chunk-strategies").json()["data"]

    values = {s["value"] for s in data["strategies"]}
    assert values == set(AVAILABLE_STRATEGIES)
    assert all(s["label"] for s in data["strategies"])
    assert "child_size" in data["current"]
    assert data["limits"]["child_size"]["min"] >= 1


def test_chunk_strategies_not_swallowed_by_doc_id_route(client: TestClient) -> None:
    """静态路径必须注册在 ``/{doc_id}`` 之前.

    FastAPI 按注册顺序匹配, "/chunk-strategies" 完全符合 "/{doc_id}" 的形状.
    顺序反了的话请求会被当成"查询 id 为 chunk-strategies 的文档"并返回 404 ——
    这类"静态路径被动态路径吃掉"的问题很常见, 必须有用例守住.
    """
    resp = client.get("/api/v1/documents/chunk-strategies")
    assert resp.status_code == 200
    assert resp.json()["code"] == "OK"


def test_list_stored_chunks(client: TestClient, sample_pdf: Path) -> None:
    doc_id = client.post(
        "/api/v1/documents",
        files={"file": (sample_pdf.name, sample_pdf.read_bytes(), "application/pdf")},
        headers={"X-User-Id": "u-chunks"},
    ).json()["data"]["document"]["id"]

    data = client.get(
        f"/api/v1/documents/{doc_id}/chunks", headers={"X-User-Id": "u-chunks"}
    ).json()["data"]

    assert data["total"] > 0
    assert data["parents"] and data["children"]
    for child in data["children"]:
        assert child["content"]
        assert child["char_count"] == len(child["content"])
        assert child["page_start"] >= 1
        assert child["type"] == "child"


def test_preview_does_not_persist(client: TestClient, sample_pdf: Path) -> None:
    """预览是只读的: 换参数试切之后, 已落库的分块不能变."""
    doc_id = client.post(
        "/api/v1/documents",
        files={"file": (sample_pdf.name, sample_pdf.read_bytes(), "application/pdf")},
        headers={"X-User-Id": "u-preview"},
    ).json()["data"]["document"]["id"]

    before = client.get(
        f"/api/v1/documents/{doc_id}/chunks", headers={"X-User-Id": "u-preview"}
    ).json()["data"]["total"]

    preview = client.post(
        f"/api/v1/documents/{doc_id}/chunk-preview",
        json={"strategy": "fixed", "parent_size": 800, "child_size": 150, "overlap": 0},
        headers={"X-User-Id": "u-preview"},
    ).json()["data"]

    assert preview["stats"]["children"] > 0
    assert preview["stats"]["params"]["strategy"] == "fixed"

    after = client.get(
        f"/api/v1/documents/{doc_id}/chunks", headers={"X-User-Id": "u-preview"}
    ).json()["data"]["total"]
    assert after == before, "预览不应改变已落库的数据"


def test_preview_returns_chunk_content(client: TestClient, sample_pdf: Path) -> None:
    """预览必须返回**实际内容**, 只给统计数字是没法判断切分好坏的."""
    doc_id = client.post(
        "/api/v1/documents",
        files={"file": (sample_pdf.name, sample_pdf.read_bytes(), "application/pdf")},
        headers={"X-User-Id": "u-preview2"},
    ).json()["data"]["document"]["id"]

    data = client.post(
        f"/api/v1/documents/{doc_id}/chunk-preview",
        json={"strategy": "parent_child", "parent_size": 600, "child_size": 150, "overlap": 0},
        headers={"X-User-Id": "u-preview2"},
    ).json()["data"]

    assert data["children"]
    for child in data["children"]:
        assert child["content"]
        assert child["parent_id"]
        assert "embedding_text" in child
        assert "embedding_differs" in child


def test_preview_rejects_invalid_params(client: TestClient, sample_pdf: Path) -> None:
    doc_id = client.post(
        "/api/v1/documents",
        files={"file": (sample_pdf.name, sample_pdf.read_bytes(), "application/pdf")},
        headers={"X-User-Id": "u-preview3"},
    ).json()["data"]["document"]["id"]

    resp = client.post(
        f"/api/v1/documents/{doc_id}/chunk-preview",
        json={"strategy": "parent_child", "parent_size": 300, "child_size": 500},
        headers={"X-User-Id": "u-preview3"},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "PARAM_INVALID"


def test_preview_isolated_between_users(client: TestClient, sample_pdf: Path) -> None:
    """预览也要做权限校验 —— 不能通过 doc_id 猜到别人的文档内容."""
    doc_id = client.post(
        "/api/v1/documents",
        files={"file": (sample_pdf.name, sample_pdf.read_bytes(), "application/pdf")},
        headers={"X-User-Id": "u-owner-cp"},
    ).json()["data"]["document"]["id"]

    resp = client.post(
        f"/api/v1/documents/{doc_id}/chunk-preview",
        json={},
        headers={"X-User-Id": "u-thief-cp"},
    )
    assert resp.status_code == 404

    assert (
        client.get(
            f"/api/v1/documents/{doc_id}/chunks", headers={"X-User-Id": "u-thief-cp"}
        ).status_code
        == 404
    )


def test_apply_chunk_params_updates_config_and_reprocesses(
    client: TestClient, sample_pdf: Path
) -> None:
    """应用参数应该同时做两件事: 存为全局配置 + 重跑当前文档.

    只存配置不重跑的话, 已有文档的分块不会变, 用户会以为"改了没生效".

    用例会真的改全局配置 —— 由模块级的 ``_restore_chunk_settings`` fixture
    负责还原, 所以这里不需要写 try/finally.
    """
    doc_id = client.post(
        "/api/v1/documents",
        files={"file": (sample_pdf.name, sample_pdf.read_bytes(), "application/pdf")},
        headers={"X-User-Id": "u-apply"},
    ).json()["data"]["document"]["id"]

    resp = client.post(
        f"/api/v1/documents/{doc_id}/chunk-apply",
        json={"strategy": "recursive", "parent_size": 1200, "child_size": 220, "overlap": 40},
        headers={"X-User-Id": "u-apply"},
    )
    assert resp.status_code == 200

    data = resp.json()["data"]
    assert data["applied_params"]["strategy"] == "recursive"
    assert data["status"] == "READY"

    # 全局配置确实被改了, 且立即生效
    assert settings.chunk_strategy == "recursive"
    assert settings.child_chunk_size == 220

    # 已落库的分块也变成了新参数的结果
    chunks = client.get(
        f"/api/v1/documents/{doc_id}/chunks", headers={"X-User-Id": "u-apply"}
    ).json()["data"]
    assert chunks["total"] > 0
