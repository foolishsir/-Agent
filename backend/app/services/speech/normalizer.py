"""朗读口语化转换.

为什么必须要有这一步
--------------------
屏幕上的文字和**能听**的文字是两种东西. 直接 ``say(markdown)`` 的效果是:

- ``**为什么**``        → 读出"星号星号为什么星号星号"
- ``[1]``               → 读出"左方括号一右方括号"
- ``→``                 → 读出"箭头"或者干脆吞掉, 造成句子断裂
- ``| 参数 | 值 |``     → 读出"竖线参数竖线值竖线"

这条在项目里很容易被漏掉 —— 因为它**只有真正戴上耳机听一遍才会发现**,
看代码是看不出来的(设计文档 docs/06 §难点2 也是这么写的).

设计原则: 只处理 TTS 一定会读错的, 不处理它本来就读对的
--------------------------------------------------------
这条反直觉但很重要. 一个常见的错误做法是把 ``40%`` 转成 ``百分之四十``、
把 ``3.5`` 转成 ``三点五`` —— 但现代 TTS(edge-tts 用的微软神经音色)
**本来就能正确读出这些**, 手工转换反而会引入新 bug:
``2.5 倍`` 被转成 ``二点五倍`` 后, 有些音色会读成"二点五倍"没错,
但 ``1/2`` 被转成 ``二分之一`` 还是 ``一比二`` 就取决于上下文了.

所以这里**只做减法**(删掉不该读的), 不做数字与单位的转换.
代价是遇到读错的数字得靠音色自己调, 收益是不会把本来对的东西改错.
"""

from __future__ import annotations

import re

#: 单次合成的文本上限兜底(字符). 面试问题都很短, 这个限制只用来挡异常输入.
DEFAULT_MAX_CHARS = 2000

# --------------------------------------------------------------------------- #
# 规则表
#
# 顺序有意义: 先删代码块(块内内容整段不要), 再删行内标记,
# 最后清理符号. 顺序反了会留下半截标记.
# --------------------------------------------------------------------------- #

# 围栏代码块整体删除 —— 代码读出来没有任何意义, 不如跳过
_RE_FENCED_CODE = re.compile(r"```[\s\S]*?```")

# 引用编号 [1] [2] / [1,2] / [1-3]
# 这是 RAG 链路的引用标记, **听的人看不到来源卡片, 读出来只是噪声**.
_RE_CITATION = re.compile(r"\[\s*\d+\s*(?:[,\-–]\s*\d+\s*)*\]")

# Markdown 链接 [文本](url) → 只留文本
_RE_MD_LINK = re.compile(r"\[([^\]]*)\]\((?:[^)]*)\)")

# 裸 URL —— 读出来是一串乱码, 直接删
_RE_URL = re.compile(r"https?://\S+")

# 行首标记: # / ## / - / * / + / > / 1. / 有序列表
_RE_LINE_PREFIX = re.compile(r"^\s{0,3}(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s*)", re.MULTILINE)

# 行内标记: **加粗** __加粗__ *斜体* _斜体_ ~~删除~~ `代码`
_RE_INLINE_MARK = re.compile(r"(\*{1,3}|_{1,3}|~~)(?=\S)(.*?)(?<=\S)\1", re.DOTALL)
_RE_INLINE_CODE = re.compile(r"`([^`]*)`")

# 表格分隔行 |---|---|
_RE_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]{3,}\|?\s*$", re.MULTILINE)

# 表情符号(含变体选择符与肤色修饰)
_RE_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f2ff"
    "\U0000fe0f\U0001f1e6-\U0001f1ff]+"
)

# 符号清理: 逐个替换成"读得出来"或"直接删掉"的
#   |  ⇒  表格残留, 逗号停顿
#   &  ⇒  "和"
#   …  ⇒  句号
_SYMBOL_MAP = {
    "⇒": "到",
    "←": "到",
    "&": "和",
    "|": "，",
    "•": "，",
    "·": "，",
    "…": "。",
    "『": "「",
    "』": "」",
}

# 数字之间的波浪号是**区间**: 「2~10 度」要读成「2 到 10 度」.
#
# 不能简单地把 ~ 删掉 —— 删掉会得到「210 度」, 意思完全变了.
# 这是"删掉标记"类规则里最容易出事的一种: 标记本身承载了语义.
_RE_NUMERIC_RANGE = re.compile(r"(?<=\d)\s*[~～]\s*(?=\d)")

# 句末标点后面的箭头是**分隔符**, 不是"到". 直接删.
# 「你做过吗？→ 说说具体怎么做的」读成「你做过吗？到 说说…」是病句.
_RE_ARROW_AFTER_SENTENCE = re.compile(r"([。！？；!?;])\s*[→⇒]\s*")

# 其余箭头当**停顿**处理, 读成逗号.
#
# 这里试过读成"到", 但在真实文本上立刻出问题:
# 「提升了 40% → 这个数字怎么测的」读成「40%到这个数字怎么测的」很别扭.
# 箭头的语义本来就分两种 —— 「A 到 B」(范围/流程)和「A, 然后 B」(转折),
# 只靠字符判断不了是哪种.
#
# 而**逗号在两种语境下都自然**:
#   「版面分析，OCR」   ✓
#   「40%，这个数字…」  ✓
# 至于真正的范围, 用户会写波浪号(2~10), 那种情况已经单独处理了.
_RE_ARROW = re.compile(r"\s*[→⇒]\s*")

# 波浪号的其他用法(非区间)直接删
_RE_TILDE = re.compile(r"[~～]")

#: 技术缩写里常见、但连着写会让 TTS 读错的东西.
#:
#: 这里**只处理少数几个**高价值替换. 大而全的缩写表是维护灾难,
#: 而且绝大多数英文缩写 TTS 读字母本来就对(A P I / Q P S / R A G).
_TECH_SPOKEN = {
    "QPS": "Q P S",
    "TPS": "T P S",
    "RTF": "R T F",
    "MRR": "M R R",
    "RRF": "R R F",
    "BM25": "B M 二十五",
    "Top-K": "Top K",
    "top-k": "top K",
    "e.g.": "例如",
    "i.e.": "也就是",
    "vs.": "对比",
    "etc.": "等等",
}


def _strip_markdown(text: str) -> str:
    """去掉 Markdown 结构, 只留可朗读的正文.

    **顺序不能随便调**, 尤其是链接必须排在裸 URL 之前:
    先删 URL 的话, ``[文本](https://...)`` 会变成 ``[文本](`` ——
    链接删了一半, 反而留下了更难看的东西.
    """
    out = _RE_FENCED_CODE.sub(" ", text)
    out = _RE_TABLE_SEP.sub("", out)
    out = _RE_MD_LINK.sub(r"\1", out)  # 必须在 _RE_URL 之前
    out = _RE_URL.sub("", out)
    out = _RE_CITATION.sub("", out)
    out = _RE_INLINE_CODE.sub(r"\1", out)
    # 行内标记跑两遍: ***粗斜体*** 这种嵌套一次剥不干净
    for _ in range(3):
        new = _RE_INLINE_MARK.sub(r"\2", out)
        if new == out:
            break
        out = new
    out = _RE_LINE_PREFIX.sub("", out)
    return _RE_EMOJI.sub("", out)


def _strip_symbols(text: str) -> str:
    # 顺序要紧: 区间 → 句末箭头 → 其余箭头 → 其余波浪号
    text = _RE_NUMERIC_RANGE.sub("到", text)
    text = _RE_ARROW_AFTER_SENTENCE.sub(r"\1", text)
    text = _RE_ARROW.sub("，", text)
    text = _RE_TILDE.sub("", text)
    for src, dst in _SYMBOL_MAP.items():
        text = text.replace(src, dst)
    return text


def _apply_spoken_terms(text: str) -> str:
    """技术缩写替换.

    用**带边界的正则**而不是 str.replace —— 否则 ``vs.`` 会命中 ``vs..``,
    ``QPS`` 会命中 ``XQPSS`` 这样的意外子串.
    """
    for term, spoken in _TECH_SPOKEN.items():
        pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"
        text = re.sub(pattern, spoken, text)
    return text


def _tidy(text: str) -> str:
    """收尾: 把标记删掉后留下的空括号、多余空白、连续标点收拾干净."""
    # 删除后残留的空括号: 「（）」「()」「【】」
    text = re.sub(r"[（(]\s*[)）]", "", text)
    text = re.sub(r"【\s*】", "", text)
    # 空白折叠
    text = re.sub(r"[ \t\u3000]+", " ", text)
    # 连续标点(删掉内容后常见): 「，。」「。。」
    text = re.sub(r"[，,]\s*(?=[，,。！？；])", "", text)
    text = re.sub(r"([。！？；])\1+", r"\1", text)
    # 标点出现在行首
    text = re.sub(r"\n\s*([，,。！？；：])", r"\1", text)
    # 行首尾空白 + 连续空行
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def to_spoken(text: str, *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """把屏幕文本转成适合朗读的口语文本.

    Args:
        text: 原始文本(可能是 Markdown, 可能带引用编号)
        max_chars: 输出长度上限; 超出会**在句子边界截断**, 而不是硬切

    Returns:
        可朗读的纯文本. 如果清理后什么都不剩(比如输入只有代码块),
        返回空串 —— 调用方应该据此跳过合成, 而不是播一段静音.
    """
    if not text:
        return ""

    out = _strip_markdown(text)
    out = _strip_symbols(out)
    out = _apply_spoken_terms(out)
    out = _tidy(out)

    if len(out) > max_chars:
        out = _truncate_at_sentence(out, max_chars)
    return out


def _truncate_at_sentence(text: str, limit: int) -> str:
    """在句子边界截断.

    硬切会把句尾的"为什么"切成"为什", 读出来是个残缺的词.
    先退到最近的句末标点, 找不到再退到逗号, 最后才硬切.

    保留比例的下限(0.4)是个折中: 卡得太严(比如要求保留一半)
    会在"第一句很短、后面是长句"的文本上失效, 退化成硬切;
    放得太松又可能只留下一个短句, 把内容砍得太多.
    """
    head = text[:limit]
    for punct in ("。", "！", "？", "；", ".", "!", "?", "\n"):
        pos = head.rfind(punct)
        if pos >= limit * 0.4:
            return head[: pos + 1]
    return head


__all__ = ["DEFAULT_MAX_CHARS", "to_spoken"]
