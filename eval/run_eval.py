"""RAG 评测运行器.

跑什么
------
1. **检索层指标**(Recall@K / Precision@K / MRR) —— 不需要 LLM, 快且免费
2. **各阶段消融** —— 向量 / 关键词 / 融合 / 去重 / 精排 / 父块扩展, 分别贡献了多少
3. **分块策略对比** —— 同一套评测集跑不同分块参数(评测集按证据文本标注, 与策略解耦)
4. **生成层指标**(可选, 需要 LLM) —— 忠实度 / 正确性 / 拒答准确率

为什么评测集按"证据文本"而不是"chunk id"标注
---------------------------------------------
chunk id 依赖分块参数, 换个 child_size 就全变了. 用 id 标注的评测集
只能验证某一个固定配置, 无法支撑"哪种分块更好"这个核心问题.
详见 ``metrics.py`` 的模块文档.

用法::

    # 单配置
    python eval/run_eval.py --pdf doc.pdf --preset chunk_parent_300

    # 全部配置对比(产出对比表)
    python eval/run_eval.py --pdf doc.pdf --all

    # 加上生成层评测(需要 LLM Key, 会花钱)
    python eval/run_eval.py --pdf doc.pdf --all --with-generation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    EVAL_USER,
    RESULTS_DIR,
    apply_runtime_config,
    describe_golden_set,
    ensure_utf8_console,
    load_golden_set,
    print_banner,
    setup_workspace,
)

ensure_utf8_console()


# --------------------------------------------------------------------------- #
# 配置预设
# --------------------------------------------------------------------------- #
@dataclass
class EvalConfig:
    """一次评测的完整配置."""

    name: str
    strategy: str = "parent_child"
    parent_size: int = 1500
    child_size: int = 300
    overlap: int = 50
    #: 覆盖默认检索参数(不填则用 .env 里的值)
    vector_top_k: int | None = None
    bm25_top_k: int | None = None
    rerank_enabled: bool | None = None
    final_top_k: int | None = None
    #: 说明, 会写进报告
    note: str = ""


PRESETS: dict[str, EvalConfig] = {
    # ---------------- 分块策略对比 ----------------
    "chunk_fixed": EvalConfig(
        name="分块: 固定长度(基线)",
        strategy="fixed",
        child_size=300,
        overlap=50,
        note="不做任何语义考量, 纯按字符硬切。作为对比基线",
    ),
    "chunk_recursive": EvalConfig(
        name="分块: 递归字符切分",
        strategy="recursive",
        child_size=300,
        overlap=50,
        note="按分隔符优先级递归, 不依赖标题识别",
    ),
    "chunk_parent_150": EvalConfig(
        name="分块: 父子块 child=150",
        strategy="parent_child",
        child_size=150,
        overlap=30,
        note="更细的检索粒度",
    ),
    "chunk_parent_300": EvalConfig(
        name="分块: 父子块 child=300",
        strategy="parent_child",
        child_size=300,
        overlap=50,
        note="默认配置",
    ),
    "chunk_parent_500": EvalConfig(
        name="分块: 父子块 child=500",
        strategy="parent_child",
        child_size=500,
        overlap=80,
        note="更粗的检索粒度",
    ),
    # ---------------- 链路消融 ----------------
    "pipeline_no_bm25": EvalConfig(
        name="链路: 关闭关键词检索",
        strategy="parent_child",
        child_size=300,
        overlap=50,
        bm25_top_k=0,
        note="退化为纯向量检索, 用于验证 BM25 那一路的贡献",
    ),
    "pipeline_no_rerank": EvalConfig(
        name="链路: 关闭精排",
        strategy="parent_child",
        child_size=300,
        overlap=50,
        rerank_enabled=False,
        note="按融合分数截断, 用于验证精排的贡献",
    ),
}

#: ``--all`` 时跑的预设(顺序即报告里的顺序)
ALL_PRESETS = [
    "chunk_fixed",
    "chunk_recursive",
    "chunk_parent_150",
    "chunk_parent_300",
    "chunk_parent_500",
    "pipeline_no_bm25",
    "pipeline_no_rerank",
]

#: 消融表中展示的阶段
STAGE_LABELS = {
    "vector_only": "① 纯向量召回",
    "bm25_only": "② 纯关键词召回",
    "rrf_fused": "③ +RRF 融合",
    "deduped": "④ +父块去重",
    "reranked": "⑤ +Cross-Encoder 精排",
    "final_parents": "⑥ 最终送入 Prompt(父块)",
}


# --------------------------------------------------------------------------- #
# 单次评测
# --------------------------------------------------------------------------- #
@dataclass
class ConfigReport:
    config: dict[str, Any]
    corpus: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    refusal: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    misses: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    per_question: list[dict[str, Any]] = field(default_factory=list)
    #: 从分数分布推导出的拒答阈值建议
    threshold_analysis: dict[str, Any] = field(default_factory=dict)


#: 被评测改动的配置项. 每跑一个配置前都要还原到基线 ——
#: 见 ``baseline_settings`` 的注释.
MUTABLE_SETTINGS = (
    "chunk_strategy",
    "parent_chunk_size",
    "child_chunk_size",
    "chunk_overlap",
    "vector_top_k",
    "bm25_top_k",
    "rerank_enabled",
    "final_top_k",
)


def baseline_settings() -> dict[str, Any]:
    """把当前配置里"评测会改动的项"快照下来.

    **为什么必须这么做**: ``settings`` 是进程级全局单例. 如果 ``run_config``
    只覆盖自己关心的项、不还原其它项, 那么上一个配置留下的值会**泄漏**到下一个.

    真实踩到的后果: ``pipeline_no_bm25`` 把 ``bm25_top_k`` 设成 0 之后,
    紧接着的 ``pipeline_no_rerank`` 没有覆盖这个字段, 于是**继承了 0** ——
    那个"关闭精排"的实验实际上是"同时关掉了关键词检索",
    结论完全不可信. 而它不会报错, 只会静静地给出一组错误的数字.

    表现上唯一的线索是结果文件名里多出了 ``nobm25`` 后缀.
    这类"配置泄漏"和测试用例之间的状态污染是同一类问题, 只是发生在评测脚本里.
    """
    from app.core.config import settings  # noqa: PLC0415

    return {name: getattr(settings, name) for name in MUTABLE_SETTINGS}


def restore_baseline(baseline: dict[str, Any]) -> None:
    """把配置还原到基线."""
    from app.core.config import settings  # noqa: PLC0415

    for name, value in baseline.items():
        setattr(settings, name, value)


def assert_applied(config: EvalConfig) -> None:
    """断言配置真的按预期生效了 —— 防住"改了没生效"这类静默失败."""
    from app.core.config import settings  # noqa: PLC0415

    expected = {
        "chunk_strategy": config.strategy,
        "parent_chunk_size": config.parent_size,
        "child_chunk_size": config.child_size,
        "chunk_overlap": config.overlap,
    }
    if config.bm25_top_k is not None:
        expected["bm25_top_k"] = config.bm25_top_k
    if config.rerank_enabled is not None:
        expected["rerank_enabled"] = config.rerank_enabled

    for name, want in expected.items():
        got = getattr(settings, name)
        if got != want:
            raise RuntimeError(f"配置未生效: {name} 期望 {want!r}, 实际 {got!r}")


async def run_config(
    config: EvalConfig,
    items: list,
    pdf: Path,
    *,
    with_generation: bool,
    k_values: tuple[int, ...],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """跑一个配置的完整评测, 返回可直接序列化的报告字典."""
    from common import ingest_pdf, reset_eval_data  # noqa: PLC0415

    from app.core.config import settings  # noqa: PLC0415
    from app.db.session import get_session_factory  # noqa: PLC0415
    from app.services.retrieval import retrieve  # noqa: PLC0415
    from eval.metrics import evaluate_refusal, evaluate_stage, latency_stats  # noqa: PLC0415

    print_banner(config.name)
    if config.note:
        print(f"说明: {config.note}")

    # ---------------- 先还原基线, 再应用本配置 ----------------
    # 顺序很重要: 先还原才能保证"本配置没显式设置的项"用的是基线值,
    # 而不是上一个配置留下的残留值.
    restore_baseline(baseline)

    settings.chunk_strategy = config.strategy
    settings.parent_chunk_size = config.parent_size
    settings.child_chunk_size = config.child_size
    settings.chunk_overlap = config.overlap
    if config.vector_top_k is not None:
        settings.vector_top_k = config.vector_top_k
    if config.bm25_top_k is not None:
        settings.bm25_top_k = config.bm25_top_k
    if config.rerank_enabled is not None:
        settings.rerank_enabled = config.rerank_enabled
    if config.final_top_k is not None:
        settings.final_top_k = config.final_top_k

    print(
        f"参数: strategy={config.strategy} parent={config.parent_size} "
        f"child={config.child_size} overlap={config.overlap} "
        f"rerank={'on' if settings.rerank_enabled else 'off'} bm25_top_k={settings.bm25_top_k}"
    )
    assert_applied(config)

    # ---------------- 清库 + 入库 ----------------
    await reset_eval_data()
    started = time.perf_counter()
    doc_id = await ingest_pdf(pdf)
    ingest_ms = (time.perf_counter() - started) * 1000

    # 统计语料规模(它就是"分块策略"最直观的产物)
    from sqlalchemy import func, select  # noqa: PLC0415

    from app.models import Chunk, ChunkType, Document  # noqa: PLC0415

    session_factory = get_session_factory()
    async with session_factory() as session:
        document = await session.get(Document, doc_id)
        parent_count = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(Chunk)
                    .where(Chunk.doc_id == doc_id, Chunk.chunk_type == ChunkType.PARENT.value)
                )
            ).scalar_one()
        )
        child_count = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(Chunk)
                    .where(Chunk.doc_id == doc_id, Chunk.chunk_type == ChunkType.CHILD.value)
                )
            ).scalar_one()
        )

    corpus = {
        "doc_id": doc_id,
        "pages": document.page_count if document else 0,
        "chars": document.char_count if document else 0,
        "parents": parent_count,
        "children": child_count,
        "ingest_ms": round(ingest_ms, 1),
    }
    print(f"入库: {corpus['parents']} 父块 / {corpus['children']} 子块, 耗时 {ingest_ms:.0f} ms\n")

    # ---------------- 逐题检索 ----------------
    stage_hits: dict[str, dict[str, list[str]]] = {k: {} for k in STAGE_LABELS}
    refused: dict[str, bool] = {}
    retrieval_ms: list[float] = []
    per_question: list[dict[str, Any]] = []
    answers: dict[str, dict[str, Any]] = {}

    for index, item in enumerate(items, start=1):
        print(f"  [{index}/{len(items)}] {item.question[:44]}", end="\r")

        async with session_factory() as session:
            t0 = time.perf_counter()
            result = await retrieve(
                session, item.question, user_id=EVAL_USER, doc_ids=[doc_id], debug=True
            )
            cost = (time.perf_counter() - t0) * 1000
        retrieval_ms.append(cost)
        refused[item.id] = result.refused

        stage_hits["vector_only"][item.id] = [
            c.content for c in (result.debug.vector_hits if result.debug else [])
        ]
        stage_hits["bm25_only"][item.id] = [
            c.content for c in (result.debug.bm25_hits if result.debug else [])
        ]
        stage_hits["rrf_fused"][item.id] = [
            c.content for c in (result.debug.fused if result.debug else [])
        ]
        stage_hits["deduped"][item.id] = [
            c.content for c in (result.debug.deduped if result.debug else [])
        ]
        stage_hits["reranked"][item.id] = [
            c.content for c in (result.debug.reranked if result.debug else [])
        ]
        stage_hits["final_parents"][item.id] = [c.content for c in result.contexts]

        per_question.append(
            {
                "id": item.id,
                "question": item.question,
                "category": item.category,
                "evidences": item.evidences,
                "refused": result.refused,
                "top_score": round(result.top_score, 4),
                "contexts": [
                    {"page": c.page_start, "section": c.section_path, "score": round(c.score, 4)}
                    for c in result.contexts
                ],
                "cost_ms": round(cost, 1),
                "debug_sizes": result.debug.sizes() if result.debug else {},
            }
        )

        if with_generation:
            answers[item.id] = await _generate_answer(session, item, doc_id)

    print(" " * 70, end="\r")

    # ---------------- 指标 ----------------
    stages = {
        key: evaluate_stage(STAGE_LABELS[key], items, stage_hits[key], k_values=k_values).to_dict()
        for key in STAGE_LABELS
    }

    report = ConfigReport(
        config={
            **dict(asdict(config)),
            "rerank_enabled": settings.rerank_enabled,
            "vector_top_k": settings.vector_top_k,
            "bm25_top_k": settings.bm25_top_k,
            "final_top_k": settings.final_top_k,
        },
        corpus=corpus,
        stages=stages,
        refusal=evaluate_refusal(items, refused),
        latency=latency_stats(retrieval_ms),
        per_question=per_question,
    )

    # 未命中样本(用于定位问题, 而不是只看一个孤零零的分数)
    for key in ("reranked", "final_parents"):
        stage_metrics = evaluate_stage(STAGE_LABELS[key], items, stage_hits[key], k_values=k_values)
        report.misses[key] = [
            {"id": s.item_id, "question": s.question, "ranks": s.evidence_ranks}
            for s in stage_metrics.misses(k_values[-1] if k_values else 5)
        ]

    # ---------------- 拒答阈值分析 ----------------
    # 用"有答案题"与"无答案题"的 Top1 分数分布来**推导**阈值,
    # 而不是凭感觉设一个数. 这是评测能产出的最实际的结论之一.
    from eval.metrics import suggest_refusal_threshold  # noqa: PLC0415

    answerable_scores = [
        row["top_score"]
        for row, item in zip(per_question, items, strict=True)
        if not item.is_no_answer
    ]
    no_answer_scores = [
        row["top_score"] for row, item in zip(per_question, items, strict=True) if item.is_no_answer
    ]
    report.threshold_analysis = suggest_refusal_threshold(answerable_scores, no_answer_scores)

    # ---------------- 生成层 ----------------
    if with_generation and answers:
        report.generation = await _grade_answers(items, answers)

    # 统一转成 dict 再返回: 下游的报告渲染、JSON 落盘都只处理普通字典,
    # 避免出现"有的地方用属性访问、有的地方用下标"这种不一致
    return asdict(report)


async def _generate_answer(session, item, doc_id: str) -> dict[str, Any]:
    """跑一次完整问答(含生成), 供生成层评测使用."""
    from app.services.rag import answer_stream  # noqa: PLC0415

    answer = ""
    contexts: list[str] = []
    refused = False
    error = ""
    first_token_ms = 0.0
    total_ms = 0.0

    async for event in answer_stream(
        session, item.question, user_id=EVAL_USER, doc_ids=[doc_id], conversation_id=None
    ):
        if event.event == "sources":
            contexts = [s["content"] for s in event.data.get("sources", [])]
        elif event.event == "done":
            answer = event.data.get("answer", "")
            refused = bool(event.data.get("refused"))
            first_token_ms = float(event.data.get("first_token_ms") or 0)
            total_ms = float(event.data.get("total_ms") or 0)
        elif event.event == "error":
            error = event.data.get("message", "")

    return {
        "answer": answer,
        "contexts": contexts,
        "refused": refused,
        "error": error,
        "first_token_ms": first_token_ms,
        "total_ms": total_ms,
    }


async def _grade_answers(items: list, answers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """用 LLM-as-judge 给生成结果打分."""
    from app.services.llm import ChatMessage, get_llm_client  # noqa: PLC0415
    from eval.metrics import (  # noqa: PLC0415
        JUDGE_SYSTEM_PROMPT,
        aggregate_generation,
        build_judge_prompt,
        latency_stats,
        parse_judge_output,
    )

    llm = get_llm_client()
    scores: list[dict[str, Any]] = []
    first_tokens: list[float] = []
    totals: list[float] = []

    print("\n生成层评测(LLM-as-judge)…")
    for index, item in enumerate(items, start=1):
        data = answers.get(item.id)
        if not data or data.get("error"):
            continue

        print(f"  评分 [{index}/{len(items)}]", end="\r")
        try:
            judged = await llm.achat(
                [
                    ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
                    ChatMessage(
                        role="user",
                        content=build_judge_prompt(
                            question=item.question,
                            answer=data["answer"],
                            reference=item.reference_answer,
                            context="\n\n---\n\n".join(data["contexts"])[:6000],
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=300,
            )
            parsed = parse_judge_output(judged.content)
        except Exception as exc:  # noqa: BLE001
            parsed = {"faithfulness": 0.0, "correctness": 0.0, "reason": f"评分失败: {exc}"}

        parsed["id"] = item.id
        parsed["category"] = item.category
        scores.append(parsed)

        if data.get("first_token_ms"):
            first_tokens.append(data["first_token_ms"])
        if data.get("total_ms"):
            totals.append(data["total_ms"])

    print(" " * 40, end="\r")
    result = aggregate_generation(scores)
    result["scores"] = scores
    result["first_token_latency"] = latency_stats(first_tokens)
    result["total_latency"] = latency_stats(totals)
    return result


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def _metric(stage: dict[str, Any], name: str, k: int | None = None) -> float:
    """从阶段指标里取值.

    ``recall_at`` / ``precision_at`` 是**嵌套字典**(每个 K 一个值),
    而 ``mrr`` 是顶层标量. 直接 ``stage.get("recall@5")`` 会永远拿到默认值 0 ——
    这正是本项目踩过的坑: 报告里 Recall 全是 0.000, 但同一张表的 MRR 却是 1.000,
    两个数字自相矛盾才让人发现读错了层级.
    """
    if name == "mrr":
        return float(stage.get("mrr", 0))
    bucket = stage.get(name) or {}
    key = f"recall@{k}" if name == "recall_at" else f"precision@{k}"
    return float(bucket.get(key, 0))


#: 递进链路上的阶段(每一步都建立在上一步之上)
CHAIN_STAGES = ["rrf_fused", "deduped", "reranked", "final_parents"]

#: 并列的单路召回基线. 它们**不是**链路的一环 ——
#: 把它们当成递进步骤会得出荒谬的结论(比如"BM25 之后 RRF 提升了 0.100",
#: 实际是把"只用 BM25"当成了 RRF 的上一阶段)
BASELINE_STAGES = ["vector_only", "bm25_only"]


def _derive_findings(reports: list[dict[str, Any]], reference: dict[str, Any], k: int) -> list[str]:
    """从指标里自动推导结论.

    为什么值得做: 一张全是数字的表, 看的人要自己找"哪个配置好、好多少".
    把结论直接写出来, 报告才能被真正使用, 而不只是存档.

    这里只陈述**数据本身**支持的结论, 不做超出数据的推断.
    """
    findings: list[str] = []
    stages = reference["stages"]

    # 1. 哪个配置的端到端排序最好
    ranked = sorted(
        reports,
        key=lambda r: _metric(r["stages"].get("final_parents", {}), "mrr"),
        reverse=True,
    )
    if len(ranked) >= 2:
        best, worst = ranked[0], ranked[-1]
        best_mrr = _metric(best["stages"].get("final_parents", {}), "mrr")
        worst_mrr = _metric(worst["stages"].get("final_parents", {}), "mrr")
        findings.append(
            f"**端到端排序最优**：`{best['config']['name']}`（MRR {best_mrr:.3f}）｜"
            f"最差：`{worst['config']['name']}`（{worst_mrr:.3f}）｜"
            f"差距 **{best_mrr - worst_mrr:+.3f}**"
        )

    # 2. 单路召回基线对比
    vector_mrr = _metric(stages.get("vector_only", {}), "mrr")
    bm25_mrr = _metric(stages.get("bm25_only", {}), "mrr")
    vector_recall = _metric(stages.get("vector_only", {}), "recall_at", k)
    bm25_recall = _metric(stages.get("bm25_only", {}), "recall_at", k)
    findings.append(
        f"**单路召回对比**（Recall@{k} / MRR）：向量 {vector_recall:.3f} / {vector_mrr:.3f}　"
        f"关键词 {bm25_recall:.3f} / {bm25_mrr:.3f}　→ "
        f"{'向量更强' if vector_mrr >= bm25_mrr else '关键词更强'}"
    )

    # 3. RRF 融合到底有没有帮上忙 —— 与**最好的单路**比, 而不是与上一阶段比
    fused_mrr = _metric(stages.get("rrf_fused", {}), "mrr")
    best_single_mrr = max(vector_mrr, bm25_mrr)
    best_single_name = "向量" if vector_mrr >= bm25_mrr else "关键词"
    if fused_mrr > best_single_mrr + 0.005:
        findings.append(
            f"✅ **RRF 融合有效**：MRR 从最好的单路（{best_single_name} "
            f"{best_single_mrr:.3f}）提升到 {fused_mrr:.3f}（{fused_mrr - best_single_mrr:+.3f}）"
            "—— 两路召回形成了互补"
        )
    elif fused_mrr < best_single_mrr - 0.005:
        findings.append(
            f"⚠️ **RRF 融合在本评测集上未带来提升**：融合后 MRR {fused_mrr:.3f} "
            f"反而低于最好的单路（{best_single_name} {best_single_mrr:.3f}，"
            f"{fused_mrr - best_single_mrr:+.3f}）。"
            "说明较弱的另一路拉低了好结果的排名。**这不代表混合检索没用** ——"
            "它在含型号/数字/专有名词的语料上价值明显（本评测集是简历，"
            "这类 token 很少），换一份设备手册类文档结论可能相反"
        )
    else:
        findings.append(
            f"**RRF 融合与最好的单路基本持平**（{fused_mrr:.3f} vs {best_single_mrr:.3f}）"
        )

    # 4. 链路各步的增量
    deltas: list[str] = []
    for i in range(1, len(CHAIN_STAGES)):
        prev, cur = CHAIN_STAGES[i - 1], CHAIN_STAGES[i]
        delta = _metric(stages.get(cur, {}), "mrr") - _metric(stages.get(prev, {}), "mrr")
        if abs(delta) >= 0.005:
            deltas.append(f"{STAGE_LABELS[cur].split(' ')[-1]} {delta:+.3f}")
    if deltas:
        findings.append("**链路各步增量**（MRR）：" + "，".join(deltas))

    # 5. 召回 vs 排序, 谁是瓶颈
    recall = _metric(stages.get("final_parents", {}), "recall_at", k)
    mrr = _metric(stages.get("final_parents", {}), "mrr")
    if recall >= 0.99 and mrr < 0.97:
        findings.append(
            f"**召回不是瓶颈**：Recall@{k} 已达 {recall:.3f}，但 MRR 只有 {mrr:.3f} —— "
            "相关内容基本都能找到，问题是**排序不够靠前**。"
            "优化方向是精排与融合策略，而不是继续加大召回条数"
        )
    elif recall < 0.9:
        findings.append(
            f"**召回是瓶颈**：Recall@{k} 仅 {recall:.3f} —— 部分问题的证据根本没进候选集。"
            "优化方向是分块粒度与 embedding 模型，调精排没有意义"
        )

    # 6. 拒答阈值的可行性
    threshold = reference.get("threshold_analysis") or {}
    answerable = threshold.get("answerable") or {}
    no_answer = threshold.get("no_answer") or {}
    if answerable and no_answer:
        high_no_answer = no_answer.get("max", 0)
        low_answerable = answerable.get("min", 1)
        if high_no_answer >= low_answerable:
            findings.append(
                f"⚠️ **精排分数阈值拒答在本评测集上失效**："
                f"无答案题最高分 {high_no_answer:.3f} ≥ 有答案题最低分 {low_answerable:.3f}，"
                "两类完全重叠。根因是精排模型判断的是「这段文字与查询是否相关」，"
                "而不是「这段文字能否回答问题」—— 话题相关但没答案的问题同样拿高分。"
                "**需要改为显式的「资料是否足以回答」判定**（先让 LLM 判可用性再决定是否生成），"
                "分数阈值只能作为粗筛"
            )
        else:
            findings.append(
                f"✅ **拒答阈值可行**：两类样本完全可分"
                f"（无答案题最高 {high_no_answer:.3f} < 有答案题最低 {low_answerable:.3f}）"
            )

    # 7. 精排的收益与代价 —— 端到端对比, 两边都取 final_parents 才是同口径
    no_rerank = next((r for r in reports if r["config"].get("rerank_enabled") is False), None)
    if no_rerank is not None:
        with_mrr = mrr
        without_mrr = _metric(no_rerank["stages"].get("final_parents", {}), "mrr")
        cost = reference["latency"].get("p95", 0) - no_rerank["latency"].get("p95", 0)
        findings.append(
            f"**精排的收益与代价**：端到端 MRR {without_mrr:.3f} → {with_mrr:.3f}"
            f"（{with_mrr - without_mrr:+.3f}），检索 P95 增加约 {cost:.0f} ms"
        )

    # 8. 分块粒度的影响
    size_reports = [
        r
        for r in reports
        if r["config"]["strategy"] == "parent_child"
        and r["config"].get("rerank_enabled") is not False
        and r["config"].get("bm25_top_k") != 0
    ]
    if len(size_reports) >= 2:
        by_size = sorted(size_reports, key=lambda r: r["config"]["child_size"])
        parts = [
            f"child={r['config']['child_size']} → {_metric(r['stages'].get('final_parents', {}), 'mrr'):.3f}"
            for r in by_size
        ]
        findings.append("**分块粒度对 MRR 的影响**：" + "，".join(parts))

    return findings


def render_report(reports: list[dict[str, Any]], golden_stats: dict[str, Any]) -> str:
    """生成 markdown 对比报告."""
    lines: list[str] = []
    k_main = 5

    lines.append("# RAG 评测报告\n")
    lines.append(
        "> 本报告由 `eval/run_eval.py` 自动生成。评测集按**证据文本**标注，"
        "与分块策略解耦，因此同一套题目可以横向对比所有参数组合。\n"
    )

    lines.append("## 评测集概况\n")
    lines.append(f"- 总题数：**{golden_stats['total']}**")
    lines.append(f"- 有答案题：{golden_stats['answerable']}")
    lines.append(f"- 无答案题（测拒答）：{golden_stats['no_answer']}")
    lines.append(f"- 分类分布：`{json.dumps(golden_stats['categories'], ensure_ascii=False)}`\n")

    # ---------------- 主表: 配置对比 ----------------
    lines.append("## 配置对比\n")
    lines.append(
        f"| 配置 | 父块 | 子块 | Recall@{k_main} | Precision@{k_main} | MRR | "
        "漏答率 | 误答率 | 检索耗时 P95 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for report in reports:
        final = report["stages"].get("final_parents", {})
        corpus = report["corpus"]
        refusal = report["refusal"]
        latency = report["latency"]
        lines.append(
            f"| {report['config']['name']} | {corpus['parents']} | {corpus['children']} | "
            f"**{_metric(final, 'recall_at', k_main):.3f}** | "
            f"{_metric(final, 'precision_at', k_main):.3f} | "
            f"{_metric(final, 'mrr'):.3f} | "
            f"{refusal.get('false_refusal_rate', 0):.3f} | "
            f"{refusal.get('false_answer_rate', 0):.3f} | "
            f"{latency.get('p95', 0):.0f} ms |"
        )
    lines.append("")

    # ---------------- 消融表 ----------------
    # 只取**一个**参考配置, 不跨配置平均.
    # 跨配置平均会把"分块差异"和"链路差异"混在一起, 让每一步的增益变得无法解释 ——
    # 比如某一步在有答案的配置里 +0.1、在另一些里 -0.05, 平均下来接近 0,
    # 看起来"这一步没用", 实际是在不同分块下效果不同. 消融实验必须控制变量.
    reference = next(
        (
            r
            for r in reports
            if r["config"]["strategy"] == "parent_child"
            and r["config"]["child_size"] == 300
            and r["config"].get("bm25_top_k") != 0
            and r["config"].get("rerank_enabled") is not False
        ),
        reports[0],
    )
    lines.append("## 链路消融：每个阶段贡献了多少\n")
    lines.append(
        f"（以 `{reference['config']['name']}` 为基准，逐阶段观察候选集的变化。"
        "消融必须控制变量，所以只看一个配置，不跨配置平均）\n"
    )
    lines.append(f"| 阶段 | 候选数 | Recall@{k_main} | MRR |")
    lines.append("|---|---|---|---|")

    # 单路基线先列出来作为参照 —— 它们不在链路上, 所以不显示增量
    for key in BASELINE_STAGES:
        stage = reference["stages"].get(key, {})
        lines.append(
            f"| {STAGE_LABELS[key]}〔单路基线〕 | {stage.get('questions', 0)} | "
            f"{_metric(stage, 'recall_at', k_main):.3f} | {_metric(stage, 'mrr'):.3f} |"
        )

    prev_mrr: float | None = None
    for key in CHAIN_STAGES:
        stage = reference["stages"].get(key, {})
        recall = _metric(stage, "recall_at", k_main)
        mrr = _metric(stage, "mrr")
        delta = ""
        if prev_mrr is not None:
            diff = mrr - prev_mrr
            delta = f" ({diff:+.3f})" if abs(diff) >= 0.001 else " (—)"
        prev_mrr = mrr
        lines.append(
            f"| {STAGE_LABELS[key]} | {stage.get('questions', 0)} | {recall:.3f} | {mrr:.3f}{delta} |"
        )
    lines.append("")
    lines.append(
        "> 前两行是**并列的单路召回基线**（不是链路的一环，所以不显示增量）。"
        "从第三行起才是递进链路，括号里是相对上一阶段的 MRR 增量。"
        "这张表回答的是「每一步到底带来了多少增益」——"
        "如果某一步的增量接近 0 甚至为负，说明它在这个语料上没起作用，"
        "应该考虑去掉或换方案，而不是继续往上叠加更多优化。\n"
    )

    # ---------------- 自动结论 ----------------
    lines.append("## 关键发现\n")
    for finding in _derive_findings(reports, reference, k_main):
        lines.append(f"- {finding}")
    lines.append("")

    # ---------------- 未命中样本 ----------------
    lines.append("## 未命中样本（定位问题用）\n")
    any_miss = False
    for report in reports:
        misses = report["misses"].get("final_parents", [])
        if not misses:
            continue
        any_miss = True
        lines.append(f"**{report['config']['name']}**")
        for miss in misses[:3]:
            lines.append(f"- `{miss['id']}` {miss['question'][:60]}")
        lines.append("")
    if not any_miss:
        lines.append("所有配置在 Recall@K 上均未出现未命中。\n")

    # ---------------- 生成层 ----------------
    graded = [r for r in reports if r.get("generation", {}).get("graded")]
    if graded:
        lines.append("## 生成层指标（LLM-as-judge）\n")
        lines.append("| 配置 | 忠实度 | 正确性 | 首字延迟 P95 | 总耗时 P95 |")
        lines.append("|---|---|---|---|---|")
        for report in graded:
            gen = report["generation"]
            lines.append(
                f"| {report['config']['name']} | {gen.get('faithfulness', 0):.3f} | "
                f"{gen.get('correctness', 0):.3f} | "
                f"{gen.get('first_token_latency', {}).get('p95', 0):.0f} ms | "
                f"{gen.get('total_latency', {}).get('p95', 0):.0f} ms |"
            )
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
async def main_async(args: argparse.Namespace) -> int:
    workspace = setup_workspace(reset=True)

    from app.db.session import init_db  # noqa: PLC0415

    applied = apply_runtime_config()
    if applied:
        print(f"已继承开发环境的 {applied} 项运行时配置(含 LLM Key)")

    await init_db()

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"[FAIL] 文档不存在: {pdf}")
        return 1

    items = load_golden_set(args.golden_set)
    if args.limit:
        items = items[: args.limit]

    golden_stats = describe_golden_set(items)
    k_values = tuple(int(k) for k in args.k_values.split(","))

    print_banner("RAG 评测")
    print(f"文档    : {pdf.name}")
    print(
        f"评测集  : {golden_stats['total']} 条 "
        f"(有答案 {golden_stats['answerable']} / 无答案 {golden_stats['no_answer']})"
    )
    print(f"工作区  : {workspace}")
    print(f"K 值    : {k_values}")

    if args.all:
        configs = [PRESETS[name] for name in ALL_PRESETS]
    elif args.preset:
        if args.preset not in PRESETS:
            print(f"[FAIL] 未知预设 {args.preset}, 可选: {', '.join(PRESETS)}")
            return 1
        configs = [PRESETS[args.preset]]
    else:
        configs = [EvalConfig(name="自定义配置")]

    baseline = baseline_settings()
    reports: list[dict[str, Any]] = []
    for config in configs:
        report = await run_config(
            config,
            items,
            pdf,
            with_generation=args.with_generation,
            k_values=k_values,
            baseline=baseline,
        )
        reports.append(report)
        _print_config_summary(report, k_values)

    # ---------------- 输出 ----------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for report in reports:
        slug = report["config"]["strategy"] + "_" + str(report["config"]["child_size"])
        if report["config"].get("bm25_top_k") == 0:
            slug += "_nobm25"
        if report["config"].get("rerank_enabled") is False:
            slug += "_norerank"
        (RESULTS_DIR / f"{slug}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    markdown = render_report(reports, golden_stats)
    report_path = RESULTS_DIR / "report.md"
    report_path.write_text(markdown, encoding="utf-8")

    print_banner("评测完成")
    print(f"明细: {RESULTS_DIR}/*.json")
    print(f"报告: {report_path}\n")
    print(markdown)
    return 0


def _print_config_summary(report: dict[str, Any], k_values: tuple[int, ...]) -> None:
    k = 5 if 5 in k_values else k_values[len(k_values) // 2]
    final = report["stages"].get("final_parents", {})
    refusal = report["refusal"]
    print(f"\n  ▸ {report['config']['name']}")
    print(
        f"    Recall@{k} = {_metric(final, 'recall_at', k):.3f}   "
        f"MRR = {_metric(final, 'mrr'):.3f}   "
        f"误答率 = {refusal.get('false_answer_rate', 0):.3f}"
    )
    print(f"    检索 P95 = {report['latency'].get('p95', 0):.0f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索/生成评测")
    parser.add_argument("--pdf", required=True, help="用于评测的文档")
    parser.add_argument("--golden-set", default=None, help="评测集路径(默认 eval/golden_set.jsonl)")
    parser.add_argument(
        "--preset", default="chunk_parent_300", help=f"配置预设: {', '.join(PRESETS)}"
    )
    parser.add_argument("--all", action="store_true", help="跑全部预设并产出对比报告")
    parser.add_argument(
        "--with-generation", action="store_true", help="同时评测生成层(需要 LLM, 会花钱)"
    )
    parser.add_argument("--k-values", default="1,3,5,10", help="统计的 K 值, 逗号分隔")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题(调试用)")
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
