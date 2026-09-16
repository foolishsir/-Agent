"""评测指标的单元测试.

指标算错比跑不出来更危险 —— 跑不出来你立刻会发现, 算错了却会拿着
错误的数字去改代码、写简历. 所以这里把每个指标都钉死.
"""

from __future__ import annotations

import pytest

from eval.metrics import (
    MATCH_THRESHOLD,
    GoldenItem,
    aggregate_generation,
    evaluate_refusal,
    evaluate_stage,
    evidence_match_ratio,
    first_match_rank,
    latency_stats,
    match_evidences,
    parse_judge_output,
    score_item,
    suggest_refusal_threshold,
)


def _item(qid: str, evidences: list[str], category: str = "semantic") -> GoldenItem:
    return GoldenItem(id=qid, question=f"问题 {qid}", evidences=evidences, category=category)


# --------------------------------------------------------------------------- #
# 证据匹配
# --------------------------------------------------------------------------- #
def test_exact_containment_scores_one() -> None:
    assert evidence_match_ratio("钢刀的更换周期为 20000 次。", "更换周期为 20000 次") == 1.0


def test_whitespace_is_ignored() -> None:
    """PDF 抽取的文本常有多余空格, "20000 次" 与 "20000次" 应视为同一段."""
    assert evidence_match_ratio("钢刀寿命20000次", "钢刀寿命 20000 次") == 1.0


def test_split_evidence_scores_partial() -> None:
    """证据被分块边界切断时, 匹配率应反映实际覆盖比例."""
    evidence = "第一句话已经完整结束。第二句话也是完整的。"
    # 只包含前半段
    ratio = evidence_match_ratio("第一句话已经完整结束。", evidence)
    assert 0.4 < ratio < 0.9


def test_unrelated_text_scores_low() -> None:
    assert evidence_match_ratio("完全无关的一段内容", "钢刀更换周期为 20000 次") < 0.3


def test_empty_inputs_score_zero() -> None:
    assert evidence_match_ratio("", "证据") == 0.0
    assert evidence_match_ratio("内容", "") == 0.0


def test_match_evidences_uses_threshold() -> None:
    evidence = "钢刀的更换周期为 20000 次"
    assert match_evidences("根据文档，钢刀的更换周期为 20000 次。", [evidence])
    assert not match_evidences("这完全是不相干的内容", [evidence])


def test_threshold_is_reasonable() -> None:
    """阈值必须在 (0, 1) 之间.

    设成 1.0 会让固定长度切分下被切断的证据全部判为未命中(所有策略指标归零);
    设成 0.5 以下又会让"恰好包含几个专有名词"的无关分块算作命中.
    """
    assert 0.5 <= MATCH_THRESHOLD <= 0.8


def test_first_match_rank() -> None:
    chunks = ["无关内容一", "无关内容二", "钢刀的更换周期为 20000 次", "无关内容四"]
    assert first_match_rank(chunks, "更换周期为 20000 次") == 3
    assert first_match_rank(chunks, "完全不存在的证据") is None


# --------------------------------------------------------------------------- #
# 单题打分
# --------------------------------------------------------------------------- #
def test_score_item_records_ranks() -> None:
    item = _item("q1", ["证据甲"])
    score = score_item(item, ["无关", "证据甲在这里", "无关"])

    assert score.evidence_ranks == [2]
    assert score.reciprocal_rank == pytest.approx(0.5)
    assert score.hit_at(2)
    assert not score.hit_at(1)


def test_multi_evidence_recall() -> None:
    item = _item(
        "q1",
        [
            "钢刀的更换周期为 20000 次或 3 个月",
            "锡膏需要在 2 到 10 摄氏度之间冷藏保存",
        ],
    )
    score = score_item(item, ["钢刀的更换周期为 20000 次或 3 个月", "无关内容"])

    assert score.recall_at(1) == pytest.approx(0.5)
    assert score.recall_at(5) == pytest.approx(0.5)
    # 只命中一条时, 排名倒数取命中的那条
    assert score.reciprocal_rank == pytest.approx(1.0)


def test_short_texts_sharing_prefix_do_not_falsely_match() -> None:
    """短文本靠比例判定会误判 —— 必须有绝对长度下限兜住.

    "证据甲" 与 "证据乙" 的最长公共子串是 "证据"(2 字), 比例 2/3 = 0.67
    已经超过 0.6 的阈值. 如果不加下限, 两个完全不同的短语会被判为互相覆盖,
    评测里就会把"未命中"算成"命中", 指标虚高.
    """
    from eval.metrics import MIN_MATCH_CHARS

    assert evidence_match_ratio("证据甲", "证据乙") == 0.0
    assert MIN_MATCH_CHARS >= 5, "下限太松挡不住短字符串的巧合子串"

    # 但对足够长的真实证据, 部分覆盖依然要能识别出来
    long_evidence = "钢刀的更换周期为 20000 次或 3 个月，到期必须强制更换"
    assert evidence_match_ratio("钢刀的更换周期为 20000 次或 3 个月", long_evidence) > 0.5


def test_precision_uses_actual_result_count() -> None:
    """分母用实际返回条数 —— 只返回 2 条时, 不该假装有 5 条并计入未命中."""
    item = _item("q1", ["证据甲"])
    score = score_item(item, ["证据甲", "无关"])

    assert score.retrieved_count == 2
    assert score.precision_at(5) == pytest.approx(0.5)


def test_empty_retrieval() -> None:
    score = score_item(_item("q1", ["证据"]), [])
    assert score.evidence_ranks == [None]
    assert score.reciprocal_rank == 0.0
    assert score.precision_at(5) == 0.0


# --------------------------------------------------------------------------- #
# 阶段聚合
# --------------------------------------------------------------------------- #
def test_evaluate_stage_excludes_no_answer_items() -> None:
    """no_answer 题没有证据可匹配, 混进检索指标会把所有指标拉低且毫无信息量."""
    items = [
        _item("q1", ["证据甲"]),
        _item("q2", [], category="no_answer"),
    ]
    metrics = evaluate_stage("测试", items, {"q1": ["证据甲"], "q2": []})

    assert metrics.questions == 2
    assert metrics.answered_questions == 1  # 只统计有证据的
    assert metrics.recall_at[1] == pytest.approx(1.0)


def test_evaluate_stage_perfect() -> None:
    items = [_item("q1", ["甲"]), _item("q2", ["乙"])]
    metrics = evaluate_stage("完美", items, {"q1": ["甲"], "q2": ["乙"]}, k_values=(1,))

    assert metrics.recall_at[1] == pytest.approx(1.0)
    assert metrics.mrr == pytest.approx(1.0)


def test_evaluate_stage_counts_empty_results() -> None:
    items = [_item("q1", ["甲"]), _item("q2", ["乙"])]
    metrics = evaluate_stage("空结果", items, {"q1": ["甲"], "q2": []})

    assert metrics.empty_results == 1


def test_evaluate_stage_dict_keys_are_nested() -> None:
    """``recall_at`` 是嵌套字典(每个 K 一个值), 不是顶层键.

    这正是报告生成里踩过的坑: 用 ``stage.get("recall@5")`` 会永远拿到 0,
    而同一张表的 MRR 却是正常值 —— 两个数字自相矛盾才暴露出读错了层级.
    """
    items = [_item("q1", ["甲"])]
    data = evaluate_stage("x", items, {"q1": ["甲"]}, k_values=(1, 5)).to_dict()

    assert data["recall_at"]["recall@5"] == pytest.approx(1.0)
    assert data["mrr"] == pytest.approx(1.0)
    assert "recall@5" not in data  # 顶层没有这个键


def test_misses_lists_unhit_questions() -> None:
    items = [_item("q1", ["甲"]), _item("q2", ["乙"])]
    metrics = evaluate_stage("x", items, {"q1": ["甲"], "q2": ["无关"]}, k_values=(1,))

    misses = metrics.misses(1)
    assert [m.item_id for m in misses] == ["q2"]


# --------------------------------------------------------------------------- #
# 拒答评估
# --------------------------------------------------------------------------- #
def test_refusal_metrics_separate_two_error_types() -> None:
    """漏答与误答必须分开统计 —— 它们的危害不对等."""
    items = [
        _item("a1", ["证据"]),
        _item("a2", ["证据"]),
        _item("n1", [], category="no_answer"),
        _item("n2", [], category="no_answer"),
    ]
    refused = {"a1": False, "a2": True, "n1": True, "n2": False}

    result = evaluate_refusal(items, refused)

    assert result["answerable"] == 2
    assert result["unanswerable"] == 2
    assert result["false_refusals"] == 1  # a2 有答案却拒答
    assert result["false_refusal_rate"] == pytest.approx(0.5)
    assert result["false_answers"] == 1  # n2 没答案却硬答
    assert result["false_answer_rate"] == pytest.approx(0.5)


def test_refusal_metrics_without_no_answer_items() -> None:
    items = [_item("a1", ["证据"])]
    result = evaluate_refusal(items, {"a1": False})

    assert result["unanswerable"] == 0
    assert result["false_answer_rate"] == 0.0


# --------------------------------------------------------------------------- #
# 阈值推导
# --------------------------------------------------------------------------- #
def test_threshold_separates_clean_data() -> None:
    result = suggest_refusal_threshold(
        answerable_scores=[0.9, 0.85, 0.95],
        no_answer_scores=[0.2, 0.3, 0.1],
    )

    assert 0.3 <= result["suggested"] <= 0.85
    assert result["est_false_answer"] == 0
    assert "完全可分" in result["rationale"]


def test_threshold_penalises_false_answers_more() -> None:
    """误答(没答案却硬答)的权重高于漏答 —— 这是产品决策, 不是技术参数."""
    # 两端各有一个样本, 阈值放在中间时两类错误各 1 个
    scores_answerable = [0.60, 0.40]
    scores_no_answer = [0.55, 0.35]

    result = suggest_refusal_threshold(scores_answerable, scores_no_answer)

    # 加权后应该偏向"多拒答"这一侧(宁可漏答也别误答)
    assert result["est_false_answer"] <= result["est_false_refusal"]


def test_threshold_reports_overlap() -> None:
    result = suggest_refusal_threshold(
        answerable_scores=[0.5, 0.9],
        no_answer_scores=[0.4, 0.8],
    )
    assert "重叠" in result["rationale"]


def test_threshold_empty_input() -> None:
    assert suggest_refusal_threshold([], []) == {}


def test_threshold_summary_stats() -> None:
    result = suggest_refusal_threshold([0.9, 0.7], [0.2])
    assert result["answerable"]["count"] == 2
    assert result["answerable"]["min"] == pytest.approx(0.7)
    assert result["no_answer"]["max"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# LLM-as-judge 输出解析
# --------------------------------------------------------------------------- #
def test_parse_plain_json() -> None:
    parsed = parse_judge_output('{"faithfulness": 0.9, "correctness": 0.8, "reason": "好"}')
    assert parsed["faithfulness"] == pytest.approx(0.9)
    assert parsed["correctness"] == pytest.approx(0.8)


def test_parse_json_in_code_block() -> None:
    """模型经常不听话地在 JSON 外面包代码块 —— 必须容错, 不能因此中断评测."""
    raw = '```json\n{"faithfulness": 1.0, "correctness": 1.0}\n```'
    assert parse_judge_output(raw)["faithfulness"] == pytest.approx(1.0)


def test_parse_json_with_preamble() -> None:
    raw = '好的，我的评分如下：{"faithfulness": 0.5, "correctness": 0.6}'
    assert parse_judge_output(raw)["correctness"] == pytest.approx(0.6)


def test_parse_clamps_out_of_range() -> None:
    parsed = parse_judge_output('{"faithfulness": 5, "correctness": -3}')
    assert parsed["faithfulness"] == 1.0
    assert parsed["correctness"] == 0.0


def test_parse_garbage_returns_zeros() -> None:
    parsed = parse_judge_output("完全不是 JSON")
    assert parsed["faithfulness"] == 0.0
    assert "无法解析" in parsed["reason"]


def test_aggregate_generation() -> None:
    result = aggregate_generation(
        [
            {"faithfulness": 1.0, "correctness": 1.0},
            {"faithfulness": 0.5, "correctness": 0.5},
        ]
    )
    assert result["graded"] == 2
    assert result["faithfulness"] == pytest.approx(0.75)


def test_aggregate_generation_empty() -> None:
    assert aggregate_generation([])["graded"] == 0


# --------------------------------------------------------------------------- #
# 延迟统计
# --------------------------------------------------------------------------- #
def test_latency_stats_uses_p95_not_mean() -> None:
    """P95 描述尾部体验 —— 平均值会被大量快请求掩盖掉少数慢请求."""
    values = [10.0] * 90 + [800.0] * 10
    stats = latency_stats(values)

    assert stats["avg"] < 120  # 平均被拉得很低
    assert stats["p95"] > 500  # 但 P95 能反映真实的慢请求
    assert stats["max"] == 800.0
    assert stats["count"] == 100


def test_latency_high_percentiles_miss_a_very_thin_tail() -> None:
    """尾部样本太少时, 高分位数也看不到它们 —— 这是分位数的数学性质, 不是 bug.

    100 个样本里只有 1 个慢请求时, P95 和 P99 都落在快请求那一侧,
    只有 ``max`` 能看到那个离群值. 把这个性质写成用例, 是为了避免
    以后有人看到"P95/P99 都正常但用户抱怨慢"时误以为统计代码写错了.
    """
    values = [10.0] * 99 + [800.0]
    stats = latency_stats(values)

    assert stats["p95"] < 100
    assert stats["p99"] < 100
    assert stats["max"] == 800.0


def test_latency_stats_empty() -> None:
    assert latency_stats([])["count"] == 0

    # 单个值也不能崩
    assert latency_stats([42.0])["p95"] == 42.0
