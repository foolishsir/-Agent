"""分块层.

对外统一入口是 ``chunk_document`` —— 它按策略分派到具体实现,
并保证三种策略产出**结构一致**的 (父块, 子块) 结果.
这样下游检索链路不需要知道用的是哪种策略, 也就不会长出分支代码.

策略对比::

    parent_child  父子块(默认/推荐). 父块按章节+长度切, 子块按句子切.
                  上下文最完整, 引用页码最准.
    recursive     按分隔符优先级递归切. 不依赖标题识别, 对结构不规整的文档更稳.
    fixed         固定长度硬切. **仅作对比实验的基线**, 会把句子从中间切断.
"""

from __future__ import annotations

from app.core.logging import get_logger, log_kv
from app.services.chunking.base import (
    Chunk,
    ChunkingResult,
    build_embedding_text,
    coalesce_short_chunks,
    make_child_id,
    make_parent_id,
)
from app.services.chunking.params import (
    AVAILABLE_STRATEGIES,
    DEFAULT_SEPARATORS,
    STRATEGY_LABELS,
    ChunkParams,
    format_separators,
    parse_separators,
    split_sentences,
)
from app.services.chunking.parent_child import chunk_parent_child
from app.services.chunking.strategies import FlatText, chunk_fixed, chunk_recursive
from app.services.parser.base import CleanDocument

logger = get_logger("docmind.chunking")


def chunk_document(
    document: CleanDocument,
    doc_id: str,
    *,
    params: ChunkParams | None = None,
    strategy: str | None = None,
    parent_size: int | None = None,
    child_size: int | None = None,
    overlap: int | None = None,
    min_size: int | None = None,
    separators: list[str] | None = None,
    keep_heading_in_child: bool | None = None,
) -> ChunkingResult:
    """按指定策略把清洗后的文档切成父子块.

    不传 ``params`` 时使用当前生效的配置(见 ``config_service``);
    单字段 kwargs 用于测试与对比实验里覆盖个别参数.

    Args:
        document: 清洗后的文档(段落自带页码)
        doc_id: 文档 id, 用于生成确定性 chunk id
        params: 完整的分块参数对象(优先级最高)
    """
    if params is None:
        params = ChunkParams.from_settings()

    # 单字段覆盖: 只在显式传了非 None 时才生效 ——
    # 如果用 `or` 处理, 传 overlap=0 (合法的"不重叠") 会被当成 falsy 而忽略
    overrides: dict[str, object] = {}
    if strategy is not None:
        overrides["strategy"] = strategy
    if parent_size is not None:
        overrides["parent_size"] = parent_size
    if child_size is not None:
        overrides["child_size"] = child_size
    if overlap is not None:
        overrides["overlap"] = overlap
    if min_size is not None:
        overrides["min_size"] = min_size
    if separators is not None:
        overrides["separators"] = separators
    if keep_heading_in_child is not None:
        overrides["keep_heading_in_child"] = keep_heading_in_child

    if overrides:
        params = _merge(params, overrides)

    params.validate()

    if params.strategy == "fixed":
        result = chunk_fixed(document, doc_id, params)
    elif params.strategy == "recursive":
        result = chunk_recursive(document, doc_id, params)
    else:
        result = chunk_parent_child(document, doc_id, params)

    log_kv(
        logger,
        "chunking.completed",
        strategy=params.strategy,
        file=document.filename,
        parents=len(result.parents),
        children=len(result.children),
    )
    return result


def _merge(base: ChunkParams, overrides: dict[str, object]) -> ChunkParams:
    """在参数对象上做局部覆盖, 返回新对象(不修改传入实例)."""
    return ChunkParams(**{**vars(base), **overrides})  # type: ignore[arg-type]


def summarize(result: ChunkingResult, params: ChunkParams | None = None) -> dict[str, object]:
    """产出分块统计, 供预览接口与界面展示.

    为什么统计值值得单独算: 调分块参数时人眼很难从几十个块里判断
    "到底哪组参数更好". 把指标摆在一起就能一眼看出问题 ——
    平均 80 字说明切太碎; 平均 200 字但 max 到 2000 说明存在异常长块.
    """
    children = result.children
    sizes = [c.char_count for c in children] or [0]

    def stats(values: list[int]) -> dict[str, float]:
        if not values:
            return {"min": 0, "max": 0, "avg": 0, "median": 0}
        ordered = sorted(values)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        return {
            "min": min(values),
            "max": max(values),
            "avg": round(sum(values) / len(values), 1),
            "median": median,
        }

    parent_chars = sum(p.char_count for p in result.parents) or 1

    summary: dict[str, object] = {
        "parents": len(result.parents),
        "children": len(children),
        "child_chars": stats(sizes),
        "parent_chars": stats([p.char_count for p in result.parents] or [0]),
        # 字符膨胀率 = 子块总字符 / 父块总字符.
        # 远大于 1 说明重叠配置过大, 块与块高度冗余 —— 检索结果里会出现大量近似重复.
        "expansion_ratio": round(sum(sizes) / parent_chars, 3),
        "heading_coverage": (
            round(sum(1 for c in children if c.section_path) / len(children), 3)
            if children
            else 0.0
        ),
    }
    if params is not None:
        summary["params"] = params.to_dict()
    return summary


__all__ = [
    "AVAILABLE_STRATEGIES",
    "DEFAULT_SEPARATORS",
    "STRATEGY_LABELS",
    "Chunk",
    "ChunkParams",
    "ChunkingResult",
    "FlatText",
    "build_embedding_text",
    "chunk_document",
    "chunk_fixed",
    "chunk_parent_child",
    "chunk_recursive",
    "coalesce_short_chunks",
    "format_separators",
    "make_child_id",
    "make_parent_id",
    "parse_separators",
    "split_sentences",
    "summarize",
]
