"""评测指标计算.

这里是**纯函数**, 不依赖数据库、模型或网络, 因此可以独立单元测试.
把指标算法和"跑评测"的编排分开, 是因为指标算错比跑不出来更危险 ——
跑不出来你会立刻发现, 算错了却会拿着错误的数字去改代码、写简历.

---

关于评测集的两种标注方式
------------------------

**方式一: 标注 chunk id**(如 ``relevant_chunk_ids: ["doc1_p0002_c000"]``)

- 优点: 精确, 判断快
- 致命缺点: **chunk id 依赖分块参数**. 换个 ``child_chunk_size`` 就是一套新 id,
  评测集立刻作废 —— 而我们评测的核心目的恰恰是"比较不同分块参数"

**方式二: 标注证据文本**(本方案)

- ``evidences: ["钢刀的更换周期为 20000 次或 3 个月"]``
- 判断方式: 计算"检索到的分块包含这段证据的比例"
- 优点: **与分块策略解耦**, 同一套评测集可以横向对比所有参数组合
- 代价: 需要模糊匹配(因为不同分块会把证据切在不同位置)

这是评测体系设计里最关键的一个决策. 用 chunk id 标注的评测集
只能验证"某个固定配置下的检索", 无法支撑参数调优 —— 而那才是评测的价值所在.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

#: 判定"这段证据被某个分块覆盖"的阈值.
#:
#: 为什么不设成 1.0(完全包含): 固定长度切分会把证据从中间切断,
#: 而恰好切在中间时, 两个分块各自覆盖约一半 —— 设成 1.0 会让所有策略都判为未命中,
#: 指标全部归零, 失去区分度.
#:
#: 为什么用 0.6 而不是 0.5: 0.5 意味着"覆盖一半就算命中", 太宽松,
#: 会让明显不相关的分块(恰好包含几个专有名词)被算成命中.
#: 60% 是实测下来区分度较好的取值 —— 完整包含的证据得 1.0,
#: 被切成两半的得 0.5 左右, 无关内容通常低于 0.3.
MATCH_THRESHOLD = 0.6

#: 匹配前归一化: 去掉所有空白.
#: 中文里"20000 次"和"20000次"应该算同一段文本, 空格不该影响匹配.
_WS_RE = re.compile(r"\s+")

#: 模糊匹配的最长证据长度. SequenceMatcher 是 O(n*m), 过长的证据会很慢.
_MAX_MATCH_LEN = 800

#: 最长公共子串的**绝对长度下限**(字符).
#:
#: 只靠比例判定会在很短的文本上误判: "证据甲" 与 "证据乙" 的最长公共子串是
#: "证据"(2 字), 比例 2/3 = 0.67 已经超过阈值, 于是两个完全不同的短语
#: 被判为互相覆盖 —— 这在评测里是致命的, 会把未命中算成命中.
#:
#: 真实证据通常是整句话(20 字以上), 加这个下限对它们没有影响;
#: 它挡住的只是短字符串之间的巧合子串.
MIN_MATCH_CHARS = 8


@dataclass
class GoldenItem:
    """评测集里的一条样本."""

    id: str
    question: str
    #: 标准答案(用于生成层评测)
    reference_answer: str = ""
    #: 必须在检索结果中出现的证据片段(用于检索层评测).
    #: 空列表表示这道题**文档中无答案**, 用于测拒答能力.
    evidences: list[str] = field(default_factory=list)
    #: exact_match | semantic | multi_hop | no_answer | multi_turn
    category: str = "semantic"
    #: 证据所在页码, 仅供人工核对
    source_pages: list[int] = field(default_factory=list)
    #: 该题属于哪份文档(多文档评测时用)
    doc_ids: list[str] = field(default_factory=list)

    @property
    def is_no_answer(self) -> bool:
        return self.category == "no_answer" or not self.evidences


def normalize(text: str) -> str:
    """匹配前归一化: 去空白 + 转小写.

    中文场景下去掉空格尤其重要: PDF 抽取的文本经常在多处插入空格,
    而"20000 次"与"20000次"在语义上完全是同一段内容.
    """
    return _WS_RE.sub("", text or "").lower()


def evidence_match_ratio(chunk_text: str, evidence: str) -> float:
    """计算一个分块对某段证据的覆盖比例, 取值 0~1.

    完全包含 → 1.0;
    被切分导致只覆盖一部分 → 按**最长公共子串**占证据的比例折算;
    无关内容 → 接近 0.

    用最长公共子串而不是字符集合重叠: 后者会把"这些字都出现过但顺序完全不同"
    的文本判为高匹配, 对中文尤其容易误判(常用汉字就那几百个).
    """
    chunk = normalize(chunk_text)
    target = normalize(evidence)
    if not chunk or not target:
        return 0.0

    # 快速路径: 完整包含
    if target in chunk:
        return 1.0

    if len(target) > _MAX_MATCH_LEN:
        target = target[:_MAX_MATCH_LEN]

    matcher = SequenceMatcher(None, chunk, target, autojunk=False)
    match = matcher.find_longest_match(0, len(chunk), 0, len(target))

    # 短于绝对下限的公共子串视为巧合, 不算命中.
    # 对短于下限的证据, 要求它几乎被完整覆盖 —— 这也是合理的:
    # 一条 3 个字的"证据"本身就不该被当作可靠的判定依据.
    if match.size < min(MIN_MATCH_CHARS, len(target)):
        return 0.0

    return match.size / len(target)


def match_evidences(chunk_text: str, evidences: list[str]) -> bool:
    """该分块是否覆盖了任意一条证据."""
    return any(evidence_match_ratio(chunk_text, e) >= MATCH_THRESHOLD for e in evidences)


def first_match_rank(chunk_texts: list[str], evidence: str) -> int | None:
    """某条证据在检索结果中首次出现的排名(从 1 开始); 未命中返回 None."""
    for rank, text in enumerate(chunk_texts, start=1):
        if evidence_match_ratio(text, evidence) >= MATCH_THRESHOLD:
            return rank
    return None


@dataclass
class QuestionScore:
    """单条问题的评分明细(保留它是为了能追查"哪道题拖低了指标")."""

    item_id: str
    question: str
    category: str
    #: 每条证据的命中排名; None 表示未命中
    evidence_ranks: list[int | None] = field(default_factory=list)
    retrieved_count: int = 0
    matched_count: int = 0

    @property
    def reciprocal_rank(self) -> float:
        ranks = [r for r in self.evidence_ranks if r is not None]
        return 1.0 / min(ranks) if ranks else 0.0

    def hit_at(self, k: int) -> bool:
        return any(r is not None and r <= k for r in self.evidence_ranks)

    def recall_at(self, k: int) -> float:
        if not self.evidence_ranks:
            return 0.0
        found = sum(1 for r in self.evidence_ranks if r is not None and r <= k)
        return found / len(self.evidence_ranks)

    def precision_at(self, k: int) -> float:
        if k <= 0:
            return 0.0
        # 分母用实际返回条数: 只有 3 条结果时, 不该假装有 5 条并计入 2 个未命中
        denom = min(k, self.retrieved_count)
        if denom == 0:
            return 0.0
        return min(self.matched_count, k) / denom


@dataclass
class StageMetrics:
    """某一阶段(或某一配置)的聚合指标."""

    name: str
    questions: int = 0
    answered_questions: int = 0  # 有证据的题数(no_answer 题不计入检索指标)
    recall_at: dict[int, float] = field(default_factory=dict)
    precision_at: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    #: 检索阶段未返回任何结果的题数
    empty_results: int = 0
    scores: list[QuestionScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "questions": self.questions,
            "answered_questions": self.answered_questions,
            "recall_at": {f"recall@{k}": round(v, 4) for k, v in sorted(self.recall_at.items())},
            "precision_at": {
                f"precision@{k}": round(v, 4) for k, v in sorted(self.precision_at.items())
            },
            "mrr": round(self.mrr, 4),
            "empty_results": self.empty_results,
        }

    def misses(self, k: int, limit: int = 5) -> list[QuestionScore]:
        """返回在 k 处未命中的题目, 用于定位问题."""
        return [s for s in self.scores if not s.hit_at(k) and s.evidence_ranks][:limit]


def evaluate_stage(
    name: str,
    items: list[GoldenItem],
    retrieved: dict[str, list[str]],
    *,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> StageMetrics:
    """计算某一阶段的检索指标.

    Args:
        items: 评测样本
        retrieved: ``{item_id: [分块文本, ...]}``, **必须按相关性从高到低排序**
        k_values: 要统计的 K 值

    Returns:
        StageMetrics. ``no_answer`` 样本**不参与**检索指标计算 ——
        它们没有证据可匹配, 混进来会把所有指标拉低, 且毫无信息量.
        它们由拒答准确率单独衡量.
    """
    metrics = StageMetrics(name=name, questions=len(items))

    ranked = [
        s
        for s in (score_item(item, retrieved.get(item.id, [])) for item in items)
        if s.evidence_ranks
    ]
    metrics.answered_questions = len(ranked)
    metrics.scores = ranked
    metrics.empty_results = sum(1 for item in items if not retrieved.get(item.id))

    if not ranked:
        return metrics

    for k in k_values:
        metrics.recall_at[k] = sum(s.recall_at(k) for s in ranked) / len(ranked)
        metrics.precision_at[k] = sum(s.precision_at(k) for s in ranked) / len(ranked)

    metrics.mrr = sum(s.reciprocal_rank for s in ranked) / len(ranked)
    return metrics


def score_item(item: GoldenItem, chunk_texts: list[str]) -> QuestionScore:
    """给单条样本打分."""
    score = QuestionScore(
        item_id=item.id,
        question=item.question,
        category=item.category,
        retrieved_count=len(chunk_texts),
        matched_count=sum(1 for t in chunk_texts if match_evidences(t, item.evidences)),
    )
    score.evidence_ranks = [first_match_rank(chunk_texts, e) for e in item.evidences]
    return score


# --------------------------------------------------------------------------- #
# 拒答准确率
# --------------------------------------------------------------------------- #
def evaluate_refusal(
    items: list[GoldenItem],
    refused: dict[str, bool],
) -> dict[str, Any]:
    """评估拒答行为.

    两个方向都要看, 而且要用**不同的指标名**区分开:

    - **漏答率**(false refusal): 有答案却没回答 → 用户体验问题, 说明阈值太严
    - **误答率**(false answer): 没答案却硬答 → **正确性问题**, 说明阈值太松, 会编造

    两者不能合成一个"准确率": 它们对产品的危害完全不同.
    误答比漏答严重得多 —— 用户宁可听到"不知道", 也不愿听到一个编造的答案.
    """
    answerable = [i for i in items if not i.is_no_answer]
    unanswerable = [i for i in items if i.is_no_answer]

    if answerable:
        false_refusals = sum(1 for i in answerable if refused.get(i.id))
        false_refusal_rate = false_refusals / len(answerable)
    else:
        false_refusals, false_refusal_rate = 0, 0.0

    if unanswerable:
        false_answers = sum(1 for i in unanswerable if not refused.get(i.id))
        false_answer_rate = false_answers / len(unanswerable)
    else:
        false_answers, false_answer_rate = 0, 0.0

    return {
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "false_refusals": false_refusals,
        "false_refusal_rate": round(false_refusal_rate, 4),
        "false_answers": false_answers,
        "false_answer_rate": round(false_answer_rate, 4),
    }


# --------------------------------------------------------------------------- #
# 生成层指标(LLM-as-judge)
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM_PROMPT = """你是一个严格的评测员，负责判断问答系统的回答质量。

只输出 JSON，不要任何解释或 markdown 代码块标记。格式：
{"faithfulness": 0.0, "correctness": 0.0, "reason": "简短理由"}

两个维度的定义：

faithfulness（忠实度，0~1）：回答中的每个论断是否都能在【参考资料】中找到依据。
- 1.0 = 全部有依据
- 0.5 = 部分有依据，部分无依据
- 0.0 = 基本是编造的
注意：**回答正确但参考资料里没有**，faithfulness 依然算 0 —— 这一维度只衡量"有没有依据"，
不衡量"内容对不对"。

correctness（正确性，0~1）：回答与【标准答案】在事实上是否一致。
- 1.0 = 事实完全一致
- 0.5 = 部分一致，或遗漏了关键信息
- 0.0 = 事实错误或答非所问
如果标准答案是"文档中未提及"，而回答也表达了无法回答，则 correctness = 1.0。"""


def build_judge_prompt(question: str, answer: str, reference: str, context: str) -> str:
    return f"""【参考资料】
{context or "（无）"}

【标准答案】
{reference or "（文档中未提及）"}

【用户问题】
{question}

【系统回答】
{answer or "（空）"}

请评分并只输出 JSON。"""


def parse_judge_output(raw: str) -> dict[str, Any]:
    """解析评测模型的输出.

    模型经常不听话地在 JSON 外面包 ```json 代码块, 或者加一句前言.
    这里做容错提取, 而不是指望它每次都严格输出 —— 评测脚本因为解析失败
    而中断, 比评分略有偏差更让人抓狂.
    """
    import json

    text = (raw or "").strip()
    # 去掉 markdown 代码块包裹
    if text.startswith("```"):
        text = re.sub(r"^```[\w]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # 直接解析失败时, 退而求其次抓第一个 JSON 对象
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"faithfulness": 0.0, "correctness": 0.0, "reason": "评测输出无法解析"}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"faithfulness": 0.0, "correctness": 0.0, "reason": "评测输出无法解析"}

    def clamp(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    return {
        "faithfulness": clamp(data.get("faithfulness", 0)),
        "correctness": clamp(data.get("correctness", 0)),
        "reason": str(data.get("reason", ""))[:200],
    }


def aggregate_generation(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合生成层指标."""
    graded = [s for s in scores if "faithfulness" in s]
    if not graded:
        return {"graded": 0, "faithfulness": 0.0, "correctness": 0.0}

    return {
        "graded": len(graded),
        "faithfulness": round(sum(s["faithfulness"] for s in graded) / len(graded), 4),
        "correctness": round(sum(s["correctness"] for s in graded) / len(graded), 4),
    }


# --------------------------------------------------------------------------- #
# 延迟统计
# --------------------------------------------------------------------------- #
def suggest_refusal_threshold(
    answerable_scores: list[float],
    no_answer_scores: list[float],
    *,
    false_answer_weight: float = 2.0,
) -> dict[str, Any]:
    """从分数分布里**推导**拒答阈值, 而不是拍脑袋定.

    拒答阈值的本质是一个二分类决策: 精排最高分 >= t 就回答, 否则拒答.
    所以它可以用"在候选阈值上扫描、取损失最小的那个"来求解 ——
    这正是分类阈值选择的常规做法.

    两类错误的权重**刻意不对等**:
    - 漏答(有答案却说不知道): 用户体验问题, 用户再问一次可能就好了
    - 误答(没答案却硬答): **正确性问题**, 用户会拿到一个看起来可信的编造内容

    所以 ``false_answer_weight`` 默认为 2.0. 这个权重不是"技术参数",
    而是产品决策 —— 应该由业务方确认, 而不是工程师默认成 1:1.

    Args:
        answerable_scores: 有答案题的 Top1 精排分
        no_answer_scores: 无答案题的 Top1 精排分

    Returns:
        建议阈值、依据, 以及按该阈值估算的漏答/误答数量
    """
    if not answerable_scores and not no_answer_scores:
        return {}

    # 候选阈值 = 所有出现过的分数(以及略高于最高分的位置)
    candidates = sorted(set(answerable_scores + no_answer_scores))
    if not candidates:
        return {}

    def loss_at(threshold: float) -> tuple[float, int, int]:
        false_refusals = sum(1 for s in answerable_scores if s < threshold)
        false_answers = sum(1 for s in no_answer_scores if s >= threshold)
        return false_refusals + false_answers * false_answer_weight, false_refusals, false_answers

    best_threshold = candidates[0]
    best_loss = float("inf")
    best_pair = (0, 0)
    for candidate in candidates:
        loss, fr, fa = loss_at(candidate)
        if loss < best_loss:
            best_loss, best_threshold, best_pair = loss, candidate, (fr, fa)

    gap = ""
    if answerable_scores and no_answer_scores:
        low_answerable = min(answerable_scores)
        high_no_answer = max(no_answer_scores)
        if high_no_answer < low_answerable:
            gap = (
                f"两类样本的分数**完全可分**(无答案题最高 {high_no_answer:.3f}, "
                f"有答案题最低 {low_answerable:.3f}), 拒答可以做到零错误"
            )
        else:
            gap = (
                f"两类样本**存在重叠区** [{high_no_answer:.3f}, {low_answerable:.3f}], "
                "无法完全分开 —— 这是检索质量的真实上限, 靠调阈值解决不了, "
                "要继续提升需要改分块或换 embedding 模型"
            )

    return {
        "suggested": round(best_threshold, 3),
        "rationale": f"在 {len(candidates)} 个候选阈值上扫描, 使加权损失最小"
        f"(误答权重 {false_answer_weight:g})。{gap}",
        "est_false_refusal": best_pair[0],
        "est_false_answer": best_pair[1],
        "answerable": _score_summary(answerable_scores),
        "no_answer": _score_summary(no_answer_scores),
    }


def _score_summary(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0}
    ordered = sorted(scores)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
        "avg": round(sum(ordered) / len(ordered), 3),
    }


def latency_stats(values: list[float]) -> dict[str, float]:
    """延迟统计. 用 P95 而不是平均值描述尾部体验.

    平均值会被大量快请求掩盖掉少数极慢的请求, 而用户记住的恰恰是那几次慢的.

    分位数用**线性插值**(numpy 的默认做法), 而不是"取第 k 个元素":
    - 取整法在样本量小的时候会跳变(99 个样本和 100 个样本可能差很多)
    - 插值法在样本量变化时更稳定, 也符合"95% 的请求快于这个值"的直觉

    ⚠️ 一个容易误解的点: 如果慢请求**恰好占 5%**, P95 会正好落在边界上,
    显示出来的仍是快请求的值 —— 这是分位数的数学性质, 不是 bug.
    想看那 5% 要盯 P99 或 max.
    """
    if not values:
        return {"count": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}

    ordered = sorted(values)

    def percentile(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = p * (len(ordered) - 1)
        lower = int(pos)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)

    return {
        "count": len(ordered),
        "avg": round(sum(ordered) / len(ordered), 1),
        "p50": round(percentile(0.50), 1),
        "p95": round(percentile(0.95), 1),
        "p99": round(percentile(0.99), 1),
        "max": round(ordered[-1], 1),
    }
