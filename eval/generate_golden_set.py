"""从文档自动生成评测集(候选), 再由人工筛选.

为什么要自动生成
----------------
手工写 30 条评测题很慢, 而且人写的问题往往会**不自觉地贴合文档措辞** ——
结果就是"用文档里的原话去搜文档", 检索看起来效果极好, 但对真实用户毫无参考价值.
让模型**只看一个分块**去生成"这个分块能回答的问题", 生成出来的问题更接近
"用户拿着一个疑问来找答案"的形态.

生成质量与人工筛选
------------------
自动生成的问题**必须人工过一遍**. 常见问题:

- 问题依赖上下文("它是什么?"), 单独拿出来无法理解
- 答案需要跨多个分块才能回答, 但只标注了一个证据
- 问题里包含了答案(等于泄题)

所以脚本输出的是**候选集**, 保存后需要人工删改. 这是刻意的:
全自动生成的评测集会给出虚高的指标, 而虚高的指标比没有指标更危险.

用法::

    python eval/generate_golden_set.py --pdf "你的文档.pdf" --per-chunk 1
    python eval/generate_golden_set.py --pdf doc.pdf --no-answer 8   # 顺便生成无答案题
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    GOLDEN_SET,
    apply_runtime_config,
    describe_golden_set,
    ensure_utf8_console,
    load_golden_set,
    print_banner,
    save_golden_set,
    setup_workspace,
)

ensure_utf8_console()
setup_workspace()
apply_runtime_config()

from eval.metrics import GoldenItem  # noqa: E402

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
GENERATE_SYSTEM = """你是一个数据集构建助手。给定一段文档片段，你要生成"用户会问的、且这段片段能够回答的问题"。

严格规则：
1. 问题必须**自包含**：不依赖任何上下文就能看懂，不要出现"它""这个""上面提到的"。
2. 问题里**不能包含答案**。比如片段说"周期为 20000 次"，就不要问"周期是 20000 次吗？"。
3. 问题要像真人提问，而不是像在复述文档标题。
4. evidence 必须是片段中**原文照抄**的一到两句连续文字，能直接支撑答案。
5. 如果这段片段信息量太少（比如只有标题、表格边框、页眉残留），返回 {"skip": true}。

只输出 JSON，不要 markdown 代码块，不要解释。格式：
{"question": "...", "reference_answer": "...", "evidence": "...", "category": "exact_match|semantic"}

category 判断：
- exact_match：包含型号、数字、日期、专有名词等需要精确匹配的内容
- semantic：需要语义理解才能找到的内容"""

GENERATE_USER = """【文档片段】
{chunk}

请生成一个问题。"""

NO_ANSWER_SYSTEM = """你是一个数据集构建助手。你要生成"看起来合理、但给定文档中**没有**答案"的问题。

规则：
1. 问题必须与文档**同一领域**（比如文档讲设备维护，就问设备维护相关但文档没写的内容）。
2. 不要问文档明显不相关的领域（比如文档讲设备维护，却问股票行情）—— 那样太容易判别。
3. 只输出 JSON 数组，不要 markdown 代码块：
["问题1", "问题2", ...]"""

NO_ANSWER_USER = """【文档概要】
以下是文档中出现过的主题关键词：
{topics}

请生成 {count} 个"同一领域但文档未涉及"的问题。"""

PARAPHRASE_SYSTEM = """你是一个数据增强助手。给定一个问题，你要把它改写成**普通用户会问的说法**。

严格规则：
1. **不能复用原问题里的专有名词和独特措辞**（人名、公司名、技术栈名称除外——那些必须保留）。
   例如原文说"上下文分层方式"，你要改成"怎么控制对话太长的问题"。
2. 不能改变问题的**意图和答案**。
3. 问题必须自包含，不依赖上下文。
4. 要像真人口语提问，而不是像在检索关键词。
5. 只输出改写后的问题本身，不要引号、不要解释、不要任何前缀。

为什么要这样改写：自动生成的问题往往照抄文档措辞，等于"用文档的原话去搜文档"，
检索必然轻松命中，指标虚高且没有区分度。口语化的改写才能反映真实用户的提问方式。"""

PARAPHRASE_USER = """【原问题】
{question}

请输出改写后的问题："""


def clean_json(raw: str) -> str:
    """去掉模型习惯性添加的 markdown 代码块包裹."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[\w]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


async def generate_for_chunk(llm, chunk_text: str, page: int) -> dict | None:
    """针对单个分块生成一条候选问题."""
    from app.services.llm import ChatMessage  # noqa: PLC0415

    try:
        result = await llm.achat(
            [
                ChatMessage(role="system", content=GENERATE_SYSTEM),
                ChatMessage(role="user", content=GENERATE_USER.format(chunk=chunk_text)),
            ],
            temperature=0.4,
            max_tokens=400,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    [WARN] 生成失败: {exc}")
        return None

    try:
        data = json.loads(clean_json(result.content))
    except json.JSONDecodeError:
        return None

    if data.get("skip"):
        return None

    question = str(data.get("question", "")).strip()
    evidence = str(data.get("evidence", "")).strip()
    if not question or not evidence:
        return None

    # 证据必须真的出自原文 —— 模型偶尔会"改写"证据, 那样匹配就失效了
    if evidence not in chunk_text:
        # 退而求其次: 用整段分块开头作为证据(人工筛选时再修)
        evidence = chunk_text.strip()[:120]

    return {
        "question": question,
        "reference_answer": str(data.get("reference_answer", "")).strip(),
        "evidence": evidence,
        "category": str(data.get("category", "semantic")).strip() or "semantic",
        "page": page,
    }


async def generate_paraphrase(llm, question: str) -> str | None:
    """把一个问题改写成口语化说法(用于提升评测集区分度)."""
    from app.services.llm import ChatMessage  # noqa: PLC0415

    try:
        result = await llm.achat(
            [
                ChatMessage(role="system", content=PARAPHRASE_SYSTEM),
                ChatMessage(role="user", content=PARAPHRASE_USER.format(question=question)),
            ],
            temperature=0.8,
            max_tokens=200,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    [WARN] 改写失败: {exc}")
        return None

    text = clean_json(result.content).strip().strip("\"\u201c\u201d'")
    # 模型偶尔会加"改写后："之类的前缀
    text = re.sub(r"^(改写后|问题|答案)[:：]\s*", "", text).strip()
    if not text or text == question:
        return None
    return text


async def generate_no_answer(llm, topics: str, count: int) -> list[str]:
    from app.services.llm import ChatMessage  # noqa: PLC0415

    try:
        result = await llm.achat(
            [
                ChatMessage(role="system", content=NO_ANSWER_SYSTEM),
                ChatMessage(
                    role="user",
                    content=NO_ANSWER_USER.format(topics=topics[:3000], count=count),
                ),
            ],
            temperature=0.7,
            max_tokens=600,
        )
        data = json.loads(clean_json(result.content))
        return [str(q).strip() for q in data if str(q).strip()][:count]
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] 无答案题生成失败: {exc}")
        return []


async def main_async(args: argparse.Namespace) -> int:
    from app.core.config import settings  # noqa: PLC0415
    from app.db.session import init_db  # noqa: PLC0415
    from app.services.llm import get_llm_client  # noqa: PLC0415

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"[FAIL] 文档不存在: {pdf}")
        return 1

    if not settings.llm_configured:
        print("[FAIL] 需要配置 DOCMIND_LLM_API_KEY 才能自动生成评测题")
        return 1

    print_banner("生成评测集候选")
    print(f"文档: {pdf.name}")

    await init_db()

    from common import ingest_pdf  # noqa: PLC0415

    from app.services.chunking import ChunkParams  # noqa: PLC0415
    from app.services.parser import clean_document, parse_pdf  # noqa: PLC0415

    doc_id = await ingest_pdf(pdf)
    print(f"已入库 | doc_id={doc_id}")

    # 直接拿解析结果按当前参数切一遍, 便于按分块生成问题
    parsed = parse_pdf(pdf)
    cleaned = clean_document(parsed)
    chunking = __import__("app.services.chunking", fromlist=["chunk_document"]).chunk_document(
        cleaned, doc_id, params=ChunkParams.from_settings()
    )

    children = [c for c in chunking.children if c.char_count >= args.min_chars]
    if not children:
        print("[FAIL] 没有足够长的分块可用于生成问题")
        return 1

    random.seed(args.seed)
    sampled = (
        random.sample(children, min(args.per_chunk_max, len(children))) if args.sample else children
    )
    print(f"从 {len(children)} 个子块中取 {len(sampled)} 个生成问题\n")

    llm = get_llm_client()
    items: list[GoldenItem] = []
    index = 0

    for child in sampled:
        for _ in range(args.per_chunk):
            index += 1
            print(f"  [{index}] 生成中… (P{child.page_start}, {child.char_count}字)", end="\r")
            data = await generate_for_chunk(llm, child.content, child.page_start)
            if not data:
                continue
            items.append(
                GoldenItem(
                    id=f"q{len(items) + 1:03d}",
                    question=data["question"],
                    reference_answer=data["reference_answer"],
                    evidences=[data["evidence"]],
                    category=data["category"],
                    source_pages=[page for page in (child.page_start, child.page_end) if page],
                    doc_ids=[doc_id],
                )
            )

    print(f"\n生成了 {len(items)} 条有答案的候选")

    # ---------------- 口语化改写(提升区分度) ----------------
    if args.paraphrase > 0:
        print(f"\n生成口语化改写(每题 {args.paraphrase} 条)…")
        derived: list[GoldenItem] = []
        for item in items:
            for _ in range(args.paraphrase):
                rewritten = await generate_paraphrase(llm, item.question)
                if not rewritten:
                    continue
                # 改写题与原题**共享同一条证据** —— 答案没变, 只是问法变了.
                # 这正是要测的能力: 用户换了说法, 系统还能不能找到同一段内容.
                derived.append(
                    GoldenItem(
                        id="pending",
                        question=rewritten,
                        reference_answer=item.reference_answer,
                        evidences=list(item.evidences),
                        category="paraphrase",
                        source_pages=list(item.source_pages),
                        doc_ids=list(item.doc_ids),
                    )
                )
        print(f"  生成了 {len(derived)} 条改写题")
        items.extend(derived)

    # ---------------- 无答案题 ----------------
    if args.no_answer > 0:
        print(f"\n生成 {args.no_answer} 条无答案题…")
        topics = "\n".join(c.content[:200] for c in sampled[:15])
        questions = await generate_no_answer(llm, topics, args.no_answer)
        for question in questions:
            items.append(
                GoldenItem(
                    id=f"q{len(items) + 1:03d}",
                    question=question,
                    reference_answer="文档中未提及",
                    evidences=[],
                    category="no_answer",
                    doc_ids=[doc_id],
                )
            )
        print(f"  生成了 {len(questions)} 条")

    if not items:
        print("[FAIL] 没有生成任何题目")
        return 1

    # ---------------- 保存 ----------------
    out_path = Path(args.out) if args.out else GOLDEN_SET
    existing: list[GoldenItem] = []
    if out_path.exists() and not args.overwrite:
        try:
            existing = load_golden_set(out_path)
            print(f"\n已有评测集 {len(existing)} 条, 将追加(用 --overwrite 覆盖)")
        except Exception:  # noqa: BLE001
            existing = []

    # id 重新编号, 避免与已有条目冲突
    merged = existing + items

    # 去重: 改写时用了较高温度, 偶尔会产出完全相同的问法.
    # 重复题会让指标被同一道题反复加权 —— 表现是"某个配置突然好了很多",
    # 实际只是那道题恰好被复制了几份.
    from eval.metrics import normalize  # noqa: PLC0415

    seen: set[str] = set()
    deduped: list[GoldenItem] = []
    for item in merged:
        key = normalize(item.question)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    removed = len(merged) - len(deduped)
    if removed:
        print(f"\n去掉了 {removed} 条重复问题")
    merged = deduped

    for i, item in enumerate(merged, start=1):
        item.id = f"q{i:03d}"

    save_golden_set(merged, out_path)

    print_banner("完成")
    stats = describe_golden_set(merged)
    print(f"评测集: {out_path}")
    print(f"  总条数    : {stats['total']}")
    print(f"  有答案    : {stats['answerable']}")
    print(f"  无答案    : {stats['no_answer']}")
    print(f"  分类分布  : {stats['categories']}")
    print()
    print("⚠️  自动生成的题目**必须人工过一遍**再使用:")
    print("   - 删掉依赖上下文、无法独立理解的问题")
    print("   - 删掉问题里已经包含答案的(等于泄题)")
    print("   - 修正 evidence: 它必须是原文照抄, 否则检索匹配会失效")
    print()
    print("建议再手工补几条 multi_hop / multi_turn 类型的题, 覆盖面更完整.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="从文档自动生成评测集候选")
    parser.add_argument("--pdf", required=True, help="用于生成问题的 PDF")
    parser.add_argument("--out", default=None, help=f"输出路径(默认 {GOLDEN_SET})")
    parser.add_argument("--per-chunk", type=int, default=1, help="每个分块生成几道题")
    parser.add_argument("--per-chunk-max", type=int, default=8, help="最多用几个分块生成")
    parser.add_argument("--min-chars", type=int, default=60, help="忽略短于该长度的分块")
    parser.add_argument("--no-answer", type=int, default=5, help="生成几条无答案题")
    parser.add_argument(
        "--paraphrase",
        type=int,
        default=0,
        help="每题额外生成几条口语化改写题。**强烈建议开启** —— 自动生成的题目"
        "往往照抄文档措辞, 等于用原话搜原文, 指标虚高且没有区分度",
    )
    parser.add_argument("--sample", action="store_true", help="随机抽样分块(默认全部)")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有评测集")
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
