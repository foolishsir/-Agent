"""引用解析与服务端校验 —— 防幻觉的**硬约束**.

为什么 Prompt 约束不够
----------------------
"请标注来源编号"这类指令只能**缓解**幻觉, 不能根治. 实测中模型会:

- 标注一个根本不存在的编号(上下文只到 [3], 它写了 [5])
- 编造出资料里没有的数字, 却挂上一个真实存在的编号
- 声称"根据 [1]"但 [1] 里根本没提这件事

前两种可以靠**代码**拦住: 编号不存在 → 直接剥离该引用.
第三种无法完全自动判定, 但可以统计"引用命中率"作为可观测指标.

因此本项目采用**四道防线**(见 docs/03):

1. 检索侧: 精排分数低于阈值 → 根本不调用 LLM
2. Prompt 侧: 强规则 + 明确拒答话术
3. 结构侧: 要求逐句标注编号
4. **服务端校验: 解析并核对编号真实性** ← 本模块

第 1 和第 4 道是代码层面的硬约束, 不依赖模型"听话". 这是工程答案与
"我靠 Prompt 防幻觉"这句初级答案的核心区别.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: 匹配 [1] 或 [1][2] 或 [1,2] 或 [1-3]
_CITATION_RE = re.compile(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]")
_RANGE_RE = re.compile(r"(\d+)\s*[\-–]\s*(\d+)")


@dataclass
class CitationCheck:
    """引用校验结果."""

    #: 答案中出现过的所有编号(去重, 保序)
    mentioned: list[int] = field(default_factory=list)
    #: 真实存在于上下文中的编号
    valid: list[int] = field(default_factory=list)
    #: 模型编造的、上下文中不存在的编号
    invalid: list[int] = field(default_factory=list)
    #: 清理后的答案(编造的编号已被剥离)
    cleaned_answer: str = ""
    #: 答案里到底有没有标注引用
    has_citation: bool = False

    @property
    def hallucinated(self) -> bool:
        """是否出现了不存在的引用编号."""
        return bool(self.invalid)

    def to_dict(self) -> dict[str, object]:
        return {
            "mentioned": self.mentioned,
            "valid": self.valid,
            "invalid": self.invalid,
            "has_citation": self.has_citation,
            "hallucinated": self.hallucinated,
        }


def parse_citation_numbers(text: str) -> list[int]:
    """从文本中解析出所有被引用的编号(展开区间, 去重保序)."""
    numbers: list[int] = []
    seen: set[int] = set()

    for match in _CITATION_RE.finditer(text):
        body = match.group(1)
        for piece in body.split(","):
            piece = piece.strip()
            if not piece:
                continue
            range_match = _RANGE_RE.fullmatch(piece)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                # 防御: 区间写反或过大时不做展开, 避免构造出巨量编号
                if 0 < start <= end <= start + 20:
                    candidates = range(start, end + 1)
                else:
                    continue
            elif piece.isdigit():
                candidates = [int(piece)]
            else:
                continue

            for number in candidates:
                if number not in seen:
                    seen.add(number)
                    numbers.append(number)

    return numbers


def validate_answer(answer: str, context_count: int) -> CitationCheck:
    """校验答案中的引用编号, 并剥离编造的引用.

    只剥离编号本身, 不删除整句话 —— 因为无法确定那句话是"基于错误引用编造的"
    还是"本来正确但编号写错了". 保留内容、去掉假引用, 是更保守的处理:
    用户看到的仍是完整回答, 只是少了一个指向不存在来源的标记.

    真正安全的做法是把"编造引用的句子"整句标红提示, 但那是产品层面的取舍,
    当前版本先保证**不给出错误的可点击引用**.
    """
    mentioned = parse_citation_numbers(answer)
    valid_range = set(range(1, context_count + 1))

    valid = [n for n in mentioned if n in valid_range]
    invalid = [n for n in mentioned if n not in valid_range]

    if invalid:
        invalid_set = set(invalid)
        answer = _CITATION_RE.sub(
            lambda m: (
                "" if any(int(x) in invalid_set for x in _split_numbers(m.group(1))) else m.group(0)
            ),
            answer,
        )
        # 剥离后可能留下多余空格和孤立的标点, 做一次轻量整理
        answer = re.sub(r"\s{2,}", " ", answer)
        answer = re.sub(r"\s+([，。；：,.!?;])", r"\1", answer)

    return CitationCheck(
        mentioned=mentioned,
        valid=valid,
        invalid=invalid,
        cleaned_answer=answer.strip(),
        has_citation=bool(valid),
    )


def _split_numbers(body: str) -> list[int]:
    """把 ``1,2-4`` 这样的编号体展开成整数列表(供替换回调使用)."""
    result: list[int] = []
    for piece in body.split(","):
        piece = piece.strip()
        range_match = _RANGE_RE.fullmatch(piece)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if 0 < start <= end <= start + 20:
                result.extend(range(start, end + 1))
        elif piece.isdigit():
            result.append(int(piece))
    return result


def build_citation_payload(contexts: list, valid_numbers: list[int]) -> list[dict[str, object]]:
    """构造返回给前端的引用列表.

    只包含**答案真正引用到的**编号 —— 把检索到的全部上下文都返回,
    会让用户以为答案参考了那么多内容, 是一种误导.
    另外按 (文档, 页码) 排序: 用户核对时是翻文档, 按页码顺序最自然.
    """
    by_index = {context.index: context for context in contexts}
    citations = []
    for number in valid_numbers:
        context = by_index.get(number)
        if context is None:
            continue
        citations.append(
            {
                "index": number,
                "doc_id": context.doc_id,
                "filename": context.filename,
                "page_start": context.page_start,
                "page_end": context.page_end,
                "section_path": context.section_path,
                "score": round(context.score, 4),
                "snippet": " ".join(context.content.split())[:150],
            }
        )

    citations.sort(key=lambda c: (str(c["filename"]), int(c["page_start"])))  # type: ignore[arg-type]
    return citations
