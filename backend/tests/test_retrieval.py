"""检索层与 RAG 层的单元测试.

这些是**纯逻辑**测试, 不需要模型也不需要网络 —— 因此可以跑得很快,
适合覆盖那些"不报错但结果错"的算法细节(RRF 排名、引用编号校验、分词).
"""

from __future__ import annotations

from app.services.rag.citation import (
    build_citation_payload,
    parse_citation_numbers,
    validate_answer,
)
from app.services.retrieval import RetrievedChunk, RetrievedContext
from app.services.retrieval.bm25 import tokenize
from app.services.retrieval.fusion import dedupe_by_parent, reciprocal_rank_fusion


# --------------------------------------------------------------------------- #
# 分词
# --------------------------------------------------------------------------- #
def test_tokenize_keeps_chinese_and_alphanumeric() -> None:
    tokens = tokenize("钢刀的更换周期是 20000 次")
    assert "钢刀" in tokens
    assert "20000" in tokens


def test_tokenize_search_mode_produces_subwords() -> None:
    """用 lcut_for_search 是为了召回率: 长词会被额外切出子词."""
    tokens = tokenize("中华人民共和国")
    # 精确模式只会给出 ["中华人民共和国"], 搜索模式还会给出子词
    assert len(tokens) > 1


def test_tokenize_drops_punctuation_only() -> None:
    tokens = tokenize("设备维护，；。！")
    assert all(t.strip() for t in tokens)
    assert "，" not in tokens


def test_tokenize_empty_text() -> None:
    assert tokenize("") == []
    assert tokenize("   ") == []


# --------------------------------------------------------------------------- #
# RRF 融合
# --------------------------------------------------------------------------- #
def _chunk(cid: str, parent: str | None = None, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid,
        content=f"内容 {cid}",
        metadata={"parent_id": parent} if parent else {},
        score=score,
        source="test",
    )


def test_rrf_boosts_chunks_found_by_both_retrievers() -> None:
    """被两路同时召回的文档应该排到最前 —— 这正是融合的价值."""
    vector_hits = [_chunk("a"), _chunk("b"), _chunk("c")]
    bm25_hits = [_chunk("c"), _chunk("d")]

    fused = reciprocal_rank_fusion([vector_hits, bm25_hits], k=60)

    # c 在向量里第 3、BM25 里第 1, a 只在向量里第 1.
    # 1/63 + 1/61 ≈ 0.0323   vs   1/61 ≈ 0.0164
    assert fused[0].id == "c"


def test_rrf_is_scale_invariant() -> None:
    """RRF 只看排名不看分数 —— 把某一路的分数放大 1000 倍不应该改变结果.

    这是它相对"加权求和"的核心优势: 两路分数量纲不同也没关系.
    """
    list_a = [_chunk("x"), _chunk("y")]
    list_b = [_chunk("x", score=999999.0), _chunk("y", score=1.0)]

    fused = reciprocal_rank_fusion([list_a, list_b])
    assert [c.id for c in fused] == ["x", "y"]


def test_rrf_deduplicates() -> None:
    fused = reciprocal_rank_fusion([[_chunk("a"), _chunk("b")], [_chunk("a")]])
    ids = [c.id for c in fused]
    assert len(ids) == len(set(ids))


def test_rrf_records_which_retrievers_found_it() -> None:
    """记录来源便于调试"为什么它排这么前"."""
    fused = reciprocal_rank_fusion([[_chunk("a")], [_chunk("a")]])
    assert len(fused[0].metadata["_rrf_sources"]) == 2


def test_rrf_handles_empty_input() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


# --------------------------------------------------------------------------- #
# 父块去重
# --------------------------------------------------------------------------- #
def test_dedupe_keeps_highest_scoring_child_per_parent() -> None:
    """同一父块被多个子块命中时只保留一个, 且保留最高分."""
    chunks = [
        _chunk("c1", parent="p1", score=0.3),
        _chunk("c2", parent="p1", score=0.9),
        _chunk("c3", parent="p2", score=0.5),
    ]
    deduped = dedupe_by_parent(chunks)

    assert len(deduped) == 2
    p1 = next(c for c in deduped if c.parent_id == "p1")
    assert p1.id == "c2"
    assert p1.score == 0.9


def test_dedupe_keeps_original_order() -> None:
    chunks = [_chunk("c1", parent="p1", score=0.1), _chunk("c2", parent="p2", score=0.9)]
    assert [c.parent_id for c in dedupe_by_parent(chunks)] == ["p1", "p2"]


def test_dedupe_uses_own_id_when_no_parent() -> None:
    """没有父块时(理论上不该发生)按自身 id 去重, 不能把它们错误合并."""
    deduped = dedupe_by_parent([_chunk("a"), _chunk("b")])
    assert len(deduped) == 2


# --------------------------------------------------------------------------- #
# 引用编号解析
# --------------------------------------------------------------------------- #
def test_parse_simple_citations() -> None:
    assert parse_citation_numbers("根据文档 [1]，钢刀寿命为 20000 次 [3]。") == [1, 3]


def test_parse_adjacent_citations() -> None:
    assert parse_citation_numbers("如 [1][2] 所述") == [1, 2]


def test_parse_comma_separated() -> None:
    assert parse_citation_numbers("参考文献 [1,3,5]") == [1, 3, 5]


def test_parse_range() -> None:
    assert parse_citation_numbers("参见 [2-4]") == [2, 3, 4]


def test_parse_deduplicates_and_keeps_order() -> None:
    assert parse_citation_numbers("[2] 和 [1] 以及 [2]") == [2, 1]


def test_parse_ignores_reversed_range() -> None:
    """防御: [5-2] 这种写反的区间不能展开出巨量编号."""
    assert parse_citation_numbers("[5-2]") == []


def test_parse_ignores_non_citation_brackets() -> None:
    assert parse_citation_numbers("这是一个[普通]方括号") == []


# --------------------------------------------------------------------------- #
# 引用校验(防幻觉硬约束)
# --------------------------------------------------------------------------- #
def test_validate_accepts_valid_citations() -> None:
    check = validate_answer("钢刀寿命为 20000 次 [1]。锡膏需冷藏 [2]。", context_count=3)

    assert check.valid == [1, 2]
    assert check.invalid == []
    assert check.hallucinated is False
    assert check.has_citation is True
    assert "[1]" in check.cleaned_answer


def test_validate_strips_hallucinated_citations() -> None:
    """模型编造了不存在的编号 → 必须剥离, 否则用户会看到一个点不开的引用.

    这是**代码层面的硬约束**, 不依赖模型"听话" —— 也是工程答案与
    "我靠 Prompt 防幻觉"这句初级答案的核心区别.
    """
    check = validate_answer("文档里说是这样 [1]，另外还有别的原因 [9]。", context_count=2)

    assert check.invalid == [9]
    assert check.hallucinated is True
    assert "[9]" not in check.cleaned_answer
    assert "[1]" in check.cleaned_answer
    # 只剥离编号, 不删句子 —— 无法判断那句话是编造的还是编号写错了
    assert "另外还有别的原因" in check.cleaned_answer


def test_validate_detects_answer_without_citation() -> None:
    check = validate_answer("钢刀寿命是三个月。", context_count=3)
    assert check.has_citation is False
    assert check.mentioned == []


def test_validate_all_invalid() -> None:
    check = validate_answer("完全无关的回答 [7][8]", context_count=2)
    assert check.invalid == [7, 8]
    assert check.valid == []


# --------------------------------------------------------------------------- #
# 引用载荷
# --------------------------------------------------------------------------- #
def _context(index: int, filename: str, page: int, score: float = 0.9) -> RetrievedContext:
    return RetrievedContext(
        index=index,
        parent_id=f"p{index}",
        doc_id="doc1",
        filename=filename,
        content=f"第 {index} 段的内容",
        page_start=page,
        page_end=page,
        section_path="第一章",
        score=score,
    )


def test_citation_payload_only_includes_referenced_ones() -> None:
    """只返回答案**真正引用到**的编号.

    把检索到的全部上下文都返回, 会让用户以为答案参考了那么多内容, 属于误导.
    """
    contexts = [_context(1, "手册.pdf", 3), _context(2, "手册.pdf", 7), _context(3, "手册.pdf", 9)]
    payload = build_citation_payload(contexts, [1, 3])

    assert [c["index"] for c in payload] == [1, 3]
    assert all(c["index"] != 2 for c in payload)


def test_citation_payload_sorted_by_page() -> None:
    """按页码排序 —— 用户核对时是翻文档, 页码顺序最自然."""
    contexts = [_context(1, "手册.pdf", 9), _context(2, "手册.pdf", 3)]
    payload = build_citation_payload(contexts, [1, 2])
    assert [c["page_start"] for c in payload] == [3, 9]


def test_citation_payload_skips_unknown_index() -> None:
    payload = build_citation_payload([_context(1, "手册.pdf", 1)], [1, 99])
    assert len(payload) == 1


def test_citation_payload_contains_page_and_snippet() -> None:
    payload = build_citation_payload([_context(1, "手册.pdf", 5)], [1])
    assert payload[0]["page_start"] == 5
    assert payload[0]["filename"] == "手册.pdf"
    assert payload[0]["snippet"]
