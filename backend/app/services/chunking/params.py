"""分块参数与策略调度.

三种策略
--------
1. **parent_child**(默认) —— 章节感知的父子块.
   父块按标题+长度切, 子块在父块内按句子边界切. 子块进向量库保检索精度,
   父块喂给模型保上下文完整.

2. **recursive** —— 按分隔符优先级递归切分(LangChain ``RecursiveCharacterTextSplitter``
   的思路). 从最"大"的分隔符(段落)开始尝试, 切出来的片段如果还是太长,
   就换更细的分隔符(换行 → 句号 → 分号 → 逗号), 最后才硬切字符.
   这是"尽量不在语义边界处切断"的通用方案, 不依赖标题识别.

3. **fixed** —— 固定长度硬切, 带重叠. 这是**基线**, 不是推荐方案.
   留着它是因为做对比实验时必须有一个"没有做任何优化"的参照物 ——
   没有基线就无法证明父子块/递归切分到底带来了多少提升.

三种策略都产出 (父块, 子块) 两层结构
------------------------------------
即使 fixed / recursive 本身不区分层级, 也会把连续的子块按大小聚合成父块.
这样做是为了让**下游检索链路保持统一** —— 检索侧的逻辑是
"子块进向量库、命中后回查父块", 如果不同策略产出的结构不一样,
检索代码就要写三种分支. 用统一结构换取下游的简单, 是划算的.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.exceptions import ParamInvalidError

#: 默认分隔符优先级(从粗到细).
#: 中文必须补上全角标点 —— 英文的 ``. `` 在中文文本里几乎不出现,
#: 直接照搬英文分隔符集合会导致中文文档完全切不开.
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",  # 段落
    "\n",  # 换行
    "。",  # 句号
    "！",
    "？",
    "；",
    "…",
    "，",  # 逗号(最后手段, 尽量不用)
)

#: 分隔符在配置/界面上的分隔记号.
#: 不能直接用逗号当分隔符的分隔符 —— 因为 ",，" 本身就是合法分隔符,
#: 会产生歧义. 用 ``|`` 就没有这个问题.
SEPARATOR_DELIMITER = "|"

AVAILABLE_STRATEGIES: tuple[str, ...] = ("parent_child", "recursive", "fixed")

STRATEGY_LABELS: dict[str, str] = {
    "parent_child": "父子块（推荐）",
    "recursive": "递归字符切分",
    "fixed": "固定长度（基线）",
}


def parse_separators(raw: str) -> list[str]:
    """把 ``\\n\\n|\\n|。|！`` 这样的配置串解析成分隔符列表."""
    if not raw:
        return list(DEFAULT_SEPARATORS)

    parts = list(raw.split(SEPARATOR_DELIMITER))
    # 还原转义写法: 界面/.env 里用户没法直接输入换行, 所以支持 \n \t 字面量
    result: list[str] = []
    for part in parts:
        token = part.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
        if token:
            result.append(token)
    return result or list(DEFAULT_SEPARATORS)


def format_separators(separators: list[str]) -> str:
    """把分隔符列表还原成可编辑的配置串(换行显示为 ``\\n`` 字面量)."""
    return SEPARATOR_DELIMITER.join(
        sep.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t") for sep in separators
    )


@dataclass
class ChunkParams:
    """一套完整的分块参数.

    独立成对象而不是散落的 kwargs: 分块参数有 6 个, 而且要能被
    "预览接口"原样传递、"应用"时持久化. 用对象承载比到处传 6 个参数清晰得多.
    """

    strategy: str = "parent_child"
    parent_size: int = 1500
    child_size: int = 300
    overlap: int = 50
    min_size: int = 30
    separators: list[str] = field(default_factory=lambda: list(DEFAULT_SEPARATORS))
    #: 是否把章节标题保留在子块正文里.
    #: True → 子块自带语境, 但引用展示时会有重复感;
    #: False → 正文干净, 靠 embedding_text 拼接章节路径来补语境(默认).
    keep_heading_in_child: bool = False

    def validate(self) -> None:
        """参数校验.

        这些约束如果不拦, **不会报错**, 只会静默产出垃圾分块 ——
        比直接抛异常难排查得多, 所以宁可在这里拒绝.
        """
        if self.strategy not in AVAILABLE_STRATEGIES:
            raise ParamInvalidError(
                f"不支持的分块策略 {self.strategy!r}, 可选: {', '.join(AVAILABLE_STRATEGIES)}"
            )
        if self.child_size >= self.parent_size:
            raise ParamInvalidError("子块大小必须小于父块大小")
        if self.overlap >= self.child_size:
            raise ParamInvalidError("重叠长度必须小于子块大小")
        # 下限设得比较宽松(20 字), 因为细粒度检索确实可能配到很小的块;
        # 真正要拦住的是"配成 0 或负数"这类明显无意义的输入.
        if min(self.parent_size, self.child_size) < 20:
            raise ParamInvalidError("分块大小不能小于 20 字")
        if not self.separators:
            raise ParamInvalidError("分隔符列表不能为空")

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "parent_size": self.parent_size,
            "child_size": self.child_size,
            "overlap": self.overlap,
            "min_size": self.min_size,
            "separators": self.separators,
            "separators_display": format_separators(self.separators),
            "keep_heading_in_child": self.keep_heading_in_child,
        }

    @classmethod
    def from_settings(cls) -> ChunkParams:
        from app.core.config import settings  # noqa: PLC0415 - 避免循环导入

        return cls(
            strategy=settings.chunk_strategy,
            parent_size=settings.parent_chunk_size,
            child_size=settings.child_chunk_size,
            overlap=settings.chunk_overlap,
            min_size=settings.min_chunk_size,
            separators=parse_separators(settings.chunk_separators),
            keep_heading_in_child=settings.chunk_keep_heading,
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ChunkParams:
        """从接口请求构造参数(只取认识的键, 未提供则用当前配置)."""
        base = cls.from_settings()

        def pick(key: str, default: object) -> object:
            value = data.get(key)
            return default if value is None or value == "" else value

        raw_separators = data.get("separators") or data.get("separators_display")
        if isinstance(raw_separators, list):
            separators = [str(s) for s in raw_separators if str(s)]
        elif isinstance(raw_separators, str):
            separators = parse_separators(raw_separators)
        else:
            separators = base.separators

        params = cls(
            strategy=str(pick("strategy", base.strategy)),
            parent_size=int(pick("parent_size", base.parent_size)),  # type: ignore[arg-type]
            child_size=int(pick("child_size", base.child_size)),  # type: ignore[arg-type]
            overlap=int(pick("overlap", base.overlap)),  # type: ignore[arg-type]
            min_size=int(pick("min_size", base.min_size)),  # type: ignore[arg-type]
            separators=separators,
            keep_heading_in_child=bool(pick("keep_heading_in_child", base.keep_heading_in_child)),
        )
        params.validate()
        return params


def split_sentences(text: str) -> list[str]:
    """按句子边界切分(中文优先).

    英文句号只在后面跟空白+大写字母时才算句末, 避免把 "3.2" "v1.0" 这类切碎.
    """
    parts = _SENTENCE_SPLIT_RE.split(re.sub(r"\n+", "\n", text))
    return [p.strip() for p in parts if p and p.strip()]


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])|(?<=\.)(?=\s+[A-Z])|\n+")
