"""问题可溯源性校验.

与 RAG 链路的**引用校验**是同一个思路, 只是校验对象从「答案」换成了「问题」:

    RAG   : 答案里的 [n] 必须能对回检索到的片段, 否则剥掉
    面试官 : 问题里的技术名词必须能在简历里找到, 否则标记为不可溯源

为什么需要
----------
模型的"知识惯性"很强. 简历里写的是 MySQL, 它可能顺口问出 "你们 Redis 集群
怎么做的" —— 这在真实面试里是**致命错误**: 候选人会立刻意识到面试官没看简历.

Prompt 里已经写了"只问简历里出现过的内容", 但**约束不能只靠 Prompt** ——
和引用校验一样, 必须在代码里再过一道.

校验策略
--------
只查**高信号词**, 不查中文常用词:
  - 拉丁字母词 (Redis / QPS / RAG / FastAPI)  —— 不在简历里就很可疑
  - 带单位的数字 (百万级 / 40% / 30ms)        —— 编造数字比编造名词更严重

中文词不做校验: "你们""怎么""为什么"这类词遍地都是, 查了全是噪声.
宁可漏报也不要误报 —— 误报会让每个问题都被标记, 标记就失去意义了.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: 拉丁字母/数字词, 长度 >= 2 —— 单个字母(x, y)基本都是变量, 不查
_LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-]{1,}")

#: 常见的技术无关英文词, 出现在问题里不代表跑题.
#:
#: 这份表决定了**误报率**, 而误报是这个功能唯一的失败模式:
#: 每个问题都被标黄, 标记就没人看了. 所以宁可漏报 ——
#: 漏报一个真跑题的名词, 顶多是没提示; 误报一个 "choose",
#: 整个标记功能立刻退化成噪声.
_STOPWORDS = frozenset(
    {
        # 功能词
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "so",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "into",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "done",
        "have",
        "has",
        "had",
        "can",
        "could",
        "will",
        "would",
        "should",
        "may",
        "might",
        "must",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        # 疑问词
        "how",
        "why",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        # 人称
        "i",
        "me",
        "my",
        "we",
        "us",
        "our",
        "you",
        "your",
        "he",
        "she",
        "they",
        "them",
        # 面试里遍地都是的通用动词/名词
        "choose",
        "chose",
        "chosen",
        "use",
        "used",
        "using",
        "make",
        "made",
        "take",
        "took",
        "get",
        "got",
        "give",
        "gave",
        "go",
        "went",
        "come",
        "came",
        "think",
        "know",
        "knew",
        "say",
        "said",
        "tell",
        "told",
        "ask",
        "asked",
        "need",
        "want",
        "like",
        "work",
        "worked",
        "works",
        "working",
        "project",
        "projects",
        "team",
        "company",
        "job",
        "role",
        "good",
        "bad",
        "better",
        "best",
        "great",
        "more",
        "most",
        "less",
        "much",
        "many",
        "some",
        "any",
        "all",
        "both",
        "each",
        "every",
        "yes",
        "no",
        "not",
        "ok",
        "okay",
        "well",
        "also",
        "just",
        "only",
        "very",
        "about",
        "after",
        "before",
        "during",
        "over",
        "under",
        "between",
        "approach",
        "method",
        "way",
        "reason",
        "point",
        "thing",
        "things",
        "part",
        "side",
        "case",
        "time",
        "times",
        "year",
        "years",
        "day",
        "days",
        "vs",
        "etc",
        "eg",
        "ie",
        "pm",
        "am",
        "ps",
    }
)

#: 纯小写且长度小于此值的词不查("he" "ok" "do" 这类基本都是功能词).
#: 含大写/数字/符号的词不受此限 —— "AI" "Go" "C#" 都是有效技术名词.
_MIN_LOWER_TOKEN_LEN = 3


@dataclass
class TraceabilityResult:
    """一次可溯源性校验的结果."""

    ok: bool
    #: 问题里出现、但简历里找不到的技术名词
    unknown_terms: list[str] = field(default_factory=list)
    #: 问题里引用到的、且在简历中确实存在的名词(用于展示"问的是简历里的东西")
    grounded_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "unknown_terms": self.unknown_terms,
            "grounded_terms": self.grounded_terms,
        }


def extract_terms(text: str) -> list[str]:
    """抽取文本里的高信号技术名词(已去重, 保留原始大小写, 结果排序稳定)."""
    seen: dict[str, str] = {}
    for match in _LATIN_TOKEN.finditer(text or ""):
        token = match.group(0).strip(".-")
        lowered = token.lower()
        if lowered in _STOPWORDS:
            continue
        # 纯小写短词基本是功能词残留, 而不是技术名词
        if token.islower() and len(token) < _MIN_LOWER_TOKEN_LEN:
            continue
        if len(token) < 2:
            continue
        seen.setdefault(lowered, token)
    # 排序保证输出稳定 —— 前端标记的位置不会因为字典序漂移而闪烁
    return [seen[k] for k in sorted(seen)]


def check_traceability(question: str, resume: str) -> TraceabilityResult:
    """校验问题是否可溯源到简历.

    Args:
        question: 模型生成的问题
        resume: 简历原文

    Returns:
        ``ok=False`` 表示问题里出现了简历中不存在的技术名词.
        **注意这不是硬失败** —— 调用方只做标记(前端标黄), 不丢弃问题:
        简历里写 "缓存中间件"、问题问 "Redis", 可能是合理的具体化,
        直接丢弃会误伤. 标记出来让人判断.
    """
    resume_lower = (resume or "").lower()
    grounded: list[str] = []
    unknown: list[str] = []

    for term in extract_terms(question):
        if term.lower() in resume_lower:
            grounded.append(term)
        else:
            unknown.append(term)

    return TraceabilityResult(
        ok=not unknown,
        unknown_terms=unknown,
        grounded_terms=grounded,
    )


__all__ = ["TraceabilityResult", "check_traceability", "extract_terms"]
