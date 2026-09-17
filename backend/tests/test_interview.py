"""面试官 Agent 的单元测试.

重点测**决策逻辑与校验逻辑**, 不测模型输出:
- 模型输出不稳定, 断言"问题应该长什么样"必然 flaky
- 而"什么时候该追问""问题是否可溯源"是**确定性的代码**, 必须锁死

这跟项目里其他测试的取舍一致: 引用校验、分块参数、拒答阈值都有测试,
而"答案写得好不好"没有测试 —— 后者本来就不该由单测保证.
"""

from __future__ import annotations

import pytest

from app.services.interview.conductor import (
    InterviewState,
    InterviewTurn,
    _clean_question,
    _decide,
    _render_evaluation,
    _render_transcript,
)
from app.services.interview.verifier import check_traceability, extract_terms

# --------------------------------------------------------------------------- #
# 追问决策: 整个面试链路里最该被锁死的一段
# --------------------------------------------------------------------------- #
DEEP = {"depth": "deep", "specificity": 0.85, "has_tradeoff": True}
SHALLOW = {"depth": "shallow", "specificity": 0.3, "has_tradeoff": False}
VAGUE = {"depth": "medium", "specificity": 0.6, "vague_words": ["大概", "好像"]}
MEDIUM = {"depth": "medium", "specificity": 0.6, "has_tradeoff": False}


@pytest.mark.parametrize(
    ("evaluation", "depth", "max_follow_up", "expected"),
    [
        # 答得又深又有取舍 → 不要为了难而难, 直接换话题
        (DEEP, 0, 3, "NEXT_TOPIC"),
        (DEEP, 2, 3, "NEXT_TOPIC"),
        # 答得浅 → 追问
        (SHALLOW, 0, 3, "FOLLOW_UP"),
        # 出现模糊词 → 追问(即使深度标称 medium)
        (VAGUE, 0, 3, "FOLLOW_UP"),
        # 中等深度: 追问一层看看能不能挖到 deep, 但不超过一层
        (MEDIUM, 0, 3, "FOLLOW_UP"),
        (MEDIUM, 1, 3, "NEXT_TOPIC"),
        # 硬约束优先: 追问层级到顶, 不管答得多浅都必须换话题
        (SHALLOW, 3, 3, "NEXT_TOPIC"),
        (VAGUE, 5, 3, "NEXT_TOPIC"),
        # max_follow_up=1 时只允许追一层
        (SHALLOW, 1, 1, "NEXT_TOPIC"),
        (SHALLOW, 0, 1, "FOLLOW_UP"),
    ],
)
def test_decide_follow_up(evaluation, depth, max_follow_up, expected):
    assert _decide(evaluation, depth, max_follow_up) == expected


def test_decide_hard_constraint_beats_quality():
    """约束必须由代码执行 —— 答得再浅, 追问层级到顶也得换话题。

    这是刻意不交给模型判断的地方: 交给它, 它一定会超。
    """
    very_shallow = {"depth": "shallow", "specificity": 0.0, "vague_words": ["不清楚"]}
    assert _decide(very_shallow, 3, 3) == "NEXT_TOPIC"


def test_decide_handles_missing_fields():
    """模型少返回字段时不能崩, 按中等质量兜底。"""
    assert _decide({}, 0, 3) in {"FOLLOW_UP", "NEXT_TOPIC"}
    assert _decide({"depth": "deep"}, 0, 3) in {"FOLLOW_UP", "NEXT_TOPIC"}


# --------------------------------------------------------------------------- #
# 问题可溯源性校验
# --------------------------------------------------------------------------- #
def test_traceability_flags_term_absent_from_resume():
    result = check_traceability(
        "你提到用 Redis 做缓存, QPS 大概多少?",
        "项目用 MySQL 做持久化, 单机跑到 3000 QPS。",
    )
    assert result.ok is False
    assert "Redis" in result.unknown_terms
    assert "QPS" in result.grounded_terms


def test_traceability_ok_when_all_terms_present():
    result = check_traceability(
        "你在这个 RAG 项目里为什么选 Chroma 而不是 FAISS?",
        "我做了个 RAG 项目, 向量库对比过 FAISS, 最后用 Chroma。",
    )
    assert result.ok is True
    assert not result.unknown_terms


def test_traceability_ignores_common_english_words():
    """and / why / choose 这类词不该被当成技术名词, 否则每个问题都会被标黄。"""
    result = check_traceability("Why did you choose this approach and how?", "简历正文")
    assert result.unknown_terms == []


def test_traceability_is_case_insensitive():
    result = check_traceability("你用的 fastapi 有什么优势?", "后端用 FastAPI 实现。")
    assert result.ok is True


def test_extract_terms_dedupes_and_keeps_length_two_plus():
    terms = extract_terms("Redis Redis a b QPS")
    assert "Redis" in terms
    assert terms.count("Redis") == 1
    # 单字母变量不查 —— 查了全是噪声
    assert "a" not in terms
    assert "b" not in terms


def test_extract_terms_keeps_symbols_in_tech_names():
    """C++ / C# / Node.js 这类带符号的名字不能被截断。"""
    terms = extract_terms("写过 C++ 和 Node.js 吗?")
    assert "C++" in terms
    assert "Node.js" in terms


# --------------------------------------------------------------------------- #
# 问题清洗
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("好的，那么你在项目里具体负责哪部分?", "你在项目里具体负责哪部分?"),
        ("接下来我想了解一下你做的缓存方案。", "我想了解一下你做的缓存方案。"),
        ("问题：你为什么要用父子分块?", "你为什么要用父子分块?"),
        ("提问: 讲讲你的检索流程", "讲讲你的检索流程"),
        ("  你做的这个项目的难点在哪?  ", "你做的这个项目的难点在哪?"),
        ('"你用过哪些中间件?"', "你用过哪些中间件?"),
    ],
)
def test_clean_question_strips_preamble(raw, expected):
    """模型很爱在问题前加铺垫, Prompt 约束过了, 代码还要再兜一层。"""
    assert _clean_question(raw) == expected


def test_clean_question_keeps_real_question_intact():
    q = "你说的 RRF 融合, k 值为什么取 60?"
    assert _clean_question(q) == q


# --------------------------------------------------------------------------- #
# Prompt 拼装
# --------------------------------------------------------------------------- #
def test_render_transcript_keeps_recent_turns_only():
    """更早的对话对提问没帮助, 只会白烧 token。"""
    turns = [
        InterviewTurn(question=f"早期问题-{i}", answer=f"早期回答-{i}") for i in range(1, 6)
    ] + [InterviewTurn(question=f"最新问题-{i}", answer=f"最新回答-{i}") for i in range(1, 4)]
    text = _render_transcript(turns, limit=3)
    assert "最新问题-3" in text
    assert "最新问题-1" in text
    assert "早期问题-5" not in text
    assert "早期问题-1" not in text


def test_render_transcript_marks_unanswered():
    turns = [InterviewTurn(question="问题", answer="")]
    assert "候选人" not in _render_transcript(turns)


def test_render_evaluation_avoids_raw_scores():
    """不要把分数丢给模型 —— 它容易过度解读 0.37 这种数字。"""
    text = _render_evaluation(
        {
            "depth": "shallow",
            "specificity": 0.37,
            "doubt": "没说清数据来源",
            "vague_words": ["大概"],
        }
    )
    assert "0.37" not in text
    assert "没说清数据来源" in text
    assert "大概" in text


def test_render_evaluation_empty_when_nothing_to_say():
    assert _render_evaluation({}) == ""


# --------------------------------------------------------------------------- #
# 状态
# --------------------------------------------------------------------------- #
def test_state_turn_count():
    state = InterviewState(doc_id="d1", skill_ids=[], resume="简历")
    assert state.turn_count == 0
    state.turns.append(InterviewTurn(question="q", answer="a"))
    assert state.turn_count == 1
