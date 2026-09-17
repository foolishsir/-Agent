"""面试官 Agent 的核心逻辑.

与 RAG 问答的本质区别
---------------------
问答链路是 ``检索 → 生成答案`` 的**直线**; 面试官是
``评估 → 决策 → 再提问`` 的**闭环状态机**:

    谁主动      : 用户 → **Agent**
    状态        : 基本无状态 → **强状态**(提纲/追问层级/轮次)
    追问        : 无 → **基于回答质量的动态决策**

两个关键决策
------------
**① 简历走全量注入, 不走检索.**
面试官需要全局视角 —— 发现经历之间的矛盾(实习写"负责后端"、项目又写
"独立完成全栈")、判断技术栈演进、规划提问顺序. 这些 Top-K 检索做不到.
简历通常 2~3 页, 全量注入完全可行.

**② 决策权在代码里, 不在模型手里.**
模型只负责"评估回答质量"并输出结构化 JSON; **是否追问由代码决定** ——
结合 ``follow_up_depth`` 与 ``max_follow_up``. 如果把这个判断也交给模型,
它一定会超(与项目里引用校验踩过的坑同源: 约束必须由代码执行).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.core.exceptions import LLMNotConfiguredError, ParamInvalidError
from app.core.logging import get_logger, log_kv
from app.services.llm import ChatMessage, get_llm_client
from app.services.skills import Skill, compose_skills

logger = get_logger("docmind.interview")

#: 简历注入的上限(字符). 超过就截断 —— 简历一般远小于这个值,
#: 设上限是为了防止有人上传几十页的作品集把上下文撑爆.
MAX_RESUME_CHARS = 20000

#: 面试提纲里最多规划几个待问点
MAX_OUTLINE_TOPICS = 8


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class InterviewTurn:
    """一轮问答."""

    question: str
    answer: str = ""
    evaluation: dict[str, Any] = field(default_factory=dict)
    decision: str = ""


@dataclass
class InterviewState:
    """面试的完整状态.

    I1 阶段这个状态由**前端持有并每次回传** —— 与问答的无状态模式一致,
    不需要在数据库里加表. 持久化留到 I3.
    """

    doc_id: str
    skill_ids: list[str]
    resume: str
    outline: list[dict[str, str]] = field(default_factory=list)
    turns: list[InterviewTurn] = field(default_factory=list)
    #: 当前话题已追问了几层
    follow_up_depth: int = 0
    #: 当前话题在提纲里的下标
    topic_index: int = 0

    @property
    def turn_count(self) -> int:
        return len(self.turns)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
PLANNER_SYSTEM = """你是一个面试提纲规划助手。

通读候选人的简历，规划出一份面试提纲。要求：

1. **只能基于简历里实际出现过的内容**，不要引入任何简历外的东西。
2. 优先挑「值得深挖」的点，而不是「好问」的点：
   - 写了量化结果（提升 40%、百万级、QPS）→ 值得追问来源与方法
   - 写了具体技术栈与中间件 → 值得追问选型与实现
   - 写了「负责」「主导」「独立完成」→ 值得确认具体做了什么
3. 一份提纲 5~8 个点就够，不要贪多。
4. 每个点包含：topic（简历里的哪个主题）、angle（从哪个角度问）、
   opening（开场问题，一句话，不要任何铺垫）。

只输出 JSON 数组，不要 markdown 代码块，不要解释：
[{"topic": "...", "angle": "...", "opening": "..."}]"""

EVALUATOR_SYSTEM = """你是一个严格的面试评估员，负责判断候选人的回答质量。

只输出 JSON，不要 markdown 代码块，不要解释：
{"depth": "shallow|medium|deep", "specificity": 0.0, "has_numbers": false,
 "has_tradeoff": false, "vague_words": [], "highlight": "", "doubt": ""}

字段含义：
- depth：技术深度。shallow=只说了"用过/了解"；medium=说了怎么做但没说为什么；
  deep=说清了原理、取舍与边界
- specificity：0~1，回答有多具体。堆砌名词得 0.2，有具体做法得 0.5，
  有具体数字/案例/时间点得 0.8 以上
- has_numbers：回答里是否出现了具体数字
- has_tradeoff：是否提到了方案取舍（为什么这么做、放弃了什么）
- vague_words：回答里的模糊词（大概、应该、差不多、好像、可能）
- highlight：回答里最亮的一点，一句话（没有则留空）
- doubt：最可疑或最含糊的一点，一句话（没有则留空）"""

QUESTION_SYSTEM = """你正在对候选人进行面试。根据当前状态生成**下一个问题**。

严格规则：
1. **只问简历里出现过的内容**。不确定某个词是否在简历里，就不要问。
2. **每次只问一个问题。** 不要把两个问题塞进一句话。
3. 不要对候选人的回答做评价，不要给正确答案。
4. 只输出问题本身，不要任何前缀、铺垫或过渡
   （不要写"好的""了解了""那么""接下来"，不要写"问题："这样的标签）。

追问时要**顺着候选人刚才的回答往下钻**，而不是换一个准备好的问题。"""


def _clean_json(text: str) -> str:
    """去掉模型习惯性添加的 markdown 代码块包裹."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[\w]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_json(text: str, *, expect: type) -> Any:
    cleaned = _clean_json(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # 退而求其次: 抓第一个 JSON 结构
        pattern = r"\[.*\]" if expect is list else r"\{.*\}"
        match = re.search(pattern, cleaned, re.DOTALL)
        if not match:
            raise ParamInvalidError("模型输出无法解析为 JSON") from None
        data = json.loads(match.group(0))

    if not isinstance(data, expect):
        raise ParamInvalidError(f"模型输出的 JSON 类型不对: 期望 {expect.__name__}")
    return data


# --------------------------------------------------------------------------- #
# 提纲规划
# --------------------------------------------------------------------------- #
async def plan_interview(resume: str, skills: list[Skill]) -> list[dict[str, str]]:
    """通读简历, 生成面试提纲.

    提纲只生成一次, 后续每轮从里面取下一个待问点 ——
    这样面试是**有主线的**, 而不是每轮临时想一个话题.
    """
    llm = get_llm_client()
    if not llm.configured:
        raise LLMNotConfiguredError()

    skill_prompt, _ = compose_skills(skills)

    result = await llm.achat(
        [
            ChatMessage(role="system", content=PLANNER_SYSTEM),
            ChatMessage(
                role="user",
                content=(
                    f"【面试风格要求】\n{skill_prompt}\n\n"
                    f"【候选人简历】\n{resume[:MAX_RESUME_CHARS]}\n\n"
                    "请规划面试提纲，只输出 JSON 数组。"
                ),
            ),
        ],
        temperature=0.4,
        max_tokens=1500,
    )

    try:
        outline = _parse_json(result.content, expect=list)
    except Exception:  # noqa: BLE001 - 提纲失败不该让整个面试起不来
        logger.exception("面试提纲解析失败, 退化为无提纲模式")
        outline = []

    cleaned: list[dict[str, str]] = []
    for item in outline[:MAX_OUTLINE_TOPICS]:
        if not isinstance(item, dict):
            continue
        opening = str(item.get("opening", "")).strip()
        if not opening:
            continue
        cleaned.append(
            {
                "topic": str(item.get("topic", "")).strip(),
                "angle": str(item.get("angle", "")).strip(),
                "opening": opening,
            }
        )

    log_kv(logger, "interview.planned", topics=len(cleaned), skills=len(skills))
    return cleaned


# --------------------------------------------------------------------------- #
# 生成下一个问题
# --------------------------------------------------------------------------- #
async def next_question(state: InterviewState, skills: list[Skill]) -> dict[str, Any]:
    """生成下一个问题, 并给出决策依据.

    决策流程(顺序不能反):

    1. 轮次到上限 → FINISH
    2. 有上一轮回答 → 让模型**评估**回答质量
    3. **代码**根据评估 + follow_up_depth 决定追问还是换话题
    4. 生成问题
    """
    llm = get_llm_client()
    if not llm.configured:
        raise LLMNotConfiguredError()

    skill_prompt, constraints = compose_skills(skills)
    max_follow_up = int(constraints["max_follow_up"])
    max_turns = int(constraints["max_turns"])

    # ---------------- ① 轮次上限 ----------------
    if state.turn_count >= max_turns:
        return {"finished": True, "reason": f"已达到轮次上限 {max_turns}", "question": ""}

    # ---------------- ② 评估上一轮回答 ----------------
    evaluation: dict[str, Any] = {}
    decision = "OPEN"
    if state.turns and state.turns[-1].answer.strip():
        evaluation = await _evaluate(state.turns[-1].question, state.turns[-1].answer)
        decision = _decide(evaluation, state.follow_up_depth, max_follow_up)

    # ---------------- ③ 生成问题 ----------------
    question = await _generate(state, skill_prompt, evaluation, decision)

    # ---------------- ④ 更新追问层级 ----------------
    if decision == "FOLLOW_UP":
        state.follow_up_depth += 1
    elif decision in ("NEXT_TOPIC", "OPEN"):
        state.follow_up_depth = 0
        if decision == "NEXT_TOPIC":
            state.topic_index += 1

    return {
        "finished": False,
        "question": question,
        "decision": decision,
        "evaluation": evaluation,
        "follow_up_depth": state.follow_up_depth,
        "topic_index": state.topic_index,
        "constraints": constraints,
    }


def _decide(evaluation: dict[str, Any], depth: int, max_follow_up: int) -> str:
    """根据评估结果决定追问还是换话题.

    **这一步放在代码里而不是交给模型**, 是刻意的:
    ``max_follow_up`` 这类约束如果交给模型自律, 它一定会超 ——
    与引用校验是同一个道理, 约束必须由代码执行.
    """
    # 硬约束优先: 追问层级到顶就强制换话题
    if depth >= max_follow_up:
        return "NEXT_TOPIC"

    answer_depth = str(evaluation.get("depth", "medium"))
    specificity = float(evaluation.get("specificity", 0.5))
    has_tradeoff = bool(evaluation.get("has_tradeoff"))
    vague = evaluation.get("vague_words") or []

    # 回答具体、有取舍 → 已经答到位了, 换话题. **不要为了难而难.**
    if answer_depth == "deep" and has_tradeoff and specificity >= 0.7:
        return "NEXT_TOPIC"

    # 含糊、浅、出现模糊词 → 往下追问
    if answer_depth == "shallow" or specificity < 0.5 or vague:
        return "FOLLOW_UP"

    # 中等深度: 追问一层看看能不能到 deep, 但不超过两级
    return "FOLLOW_UP" if depth < 1 else "NEXT_TOPIC"


async def _evaluate(question: str, answer: str) -> dict[str, Any]:
    """评估一轮回答的质量."""
    llm = get_llm_client()
    try:
        result = await llm.achat(
            [
                ChatMessage(role="system", content=EVALUATOR_SYSTEM),
                ChatMessage(role="user", content=f"【问题】\n{question}\n\n【回答】\n{answer}"),
            ],
            temperature=0.0,
            max_tokens=400,
        )
        return _parse_json(result.content, expect=dict)
    except Exception:  # noqa: BLE001 - 评估失败不该中断面试
        logger.exception("回答评估失败, 按中等质量处理")
        return {
            "depth": "medium",
            "specificity": 0.5,
            "has_numbers": False,
            "has_tradeoff": False,
            "vague_words": [],
            "highlight": "",
            "doubt": "",
        }


async def _generate(
    state: InterviewState,
    skill_prompt: str,
    evaluation: dict[str, Any],
    decision: str,
) -> str:
    """生成下一个问题."""
    llm = get_llm_client()

    # 当前话题: 优先用提纲里规划好的点
    topic = ""
    if decision in ("OPEN", "NEXT_TOPIC") and state.topic_index < len(state.outline):
        item = state.outline[state.topic_index]
        topic = f"【当前话题】{item.get('topic')}｜切入角度：{item.get('angle')}"

    transcript = _render_transcript(state.turns)
    directive = {
        "OPEN": "这是面试的第一个问题。直接抛出你的开场问题。",
        "FOLLOW_UP": (
            "候选人刚才的回答还不够具体，请**顺着他的回答继续追问**，"
            "把他没讲清楚的那一点挖出来。不要换话题。"
        ),
        "NEXT_TOPIC": "这个话题已经问到位了，请换到下一个话题。",
    }[decision]

    user_prompt = f"""【面试风格】
{skill_prompt}

【候选人简历】
{state.resume[:MAX_RESUME_CHARS]}

{_render_outline(state)}

{topic}

【对话记录】
{transcript or "（面试刚开始）"}

【本轮指令】
{directive}
{_render_evaluation(evaluation) if evaluation else ""}

请只输出下一个问题本身。"""

    result = await llm.achat(
        [
            ChatMessage(role="system", content=QUESTION_SYSTEM),
            ChatMessage(role="user", content=user_prompt),
        ],
        temperature=0.7,
        max_tokens=300,
    )

    return _clean_question(result.content)


def _render_transcript(turns: list[InterviewTurn], limit: int = 6) -> str:
    """渲染最近的对话记录. 只保留最近几轮 —— 更早的对话对提问没有帮助, 只是浪费 token."""
    lines: list[str] = []
    for turn in turns[-limit:]:
        lines.append(f"面试官：{turn.question}")
        if turn.answer:
            lines.append(f"候选人：{turn.answer}")
    return "\n\n".join(lines)


def _render_outline(state: InterviewState) -> str:
    if not state.outline:
        return ""
    items = "\n".join(
        f"  {i + 1}. {item.get('topic')}（{item.get('angle')}）"
        + ("  ← 当前" if i == state.topic_index else "")
        for i, item in enumerate(state.outline)
    )
    return f"【面试提纲】\n{items}"


def _render_evaluation(evaluation: dict[str, Any]) -> str:
    """把评估结果转成给模型的提示.

    注意**只传判断结论, 不传具体分数** —— 把 0.37 这种数字丢给模型,
    它容易过度解读. 告诉它"这个回答偏浅、有模糊词"就够了.
    """
    hints: list[str] = []
    if evaluation.get("doubt"):
        hints.append(f"可疑点：{evaluation['doubt']}")
    if evaluation.get("highlight"):
        hints.append(f"亮点：{evaluation['highlight']}")
    if evaluation.get("vague_words"):
        hints.append(f"出现了模糊表述：{'、'.join(map(str, evaluation['vague_words']))}")
    if not hints:
        return ""
    return "\n【上一轮回答的评估】\n" + "\n".join(f"  - {h}" for h in hints)


def _clean_question(text: str) -> str:
    """清洗模型输出的问题.

    模型很容易在问题前加"好的，了解了。那么接下来我想问一下…"这类铺垫,
    读起来特别啰嗦. SKILL 里已经约束过, 但**约束不能只靠 Prompt**,
    这里再做一层后处理兜底.

    注意是**循环**剥离, 不是一次: "好的，那么你…" 里的两层前缀
    要各剥一次. 剥到不再变化为止.
    """
    question = (text or "").strip().strip('"').strip()

    prefixes = re.compile(
        r"^(好的|好|了解了|了解|明白了|明白|嗯|呃|那么|接下来|下一个问题|下面|"
        r"问题|提问|我想问一下|我想问|请问)[，,。.、：:；;\s]*"
    )
    # 加一个次数上限, 防止正则意外匹配空串导致死循环
    for _ in range(6):
        cleaned = prefixes.sub("", question).strip()
        if cleaned == question:
            break
        question = cleaned

    return question.strip()


__all__ = [
    "MAX_RESUME_CHARS",
    "InterviewState",
    "InterviewTurn",
    "next_question",
    "plan_interview",
]
