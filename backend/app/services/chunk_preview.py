"""分块查看与预览服务.

解决什么问题
------------
分块策略是 RAG 效果的**上限** —— 切坏了, 后面 Prompt 写得再好也救不回来.
但"调分块参数"在大多数项目里是非常痛苦的过程: 改一次配置 → 重新上传 →
重新解析 → 重新向量化 → 再看结果. 一轮几分钟, 试三组参数半小时就过去了.

这里提供**实时预览**: 用当前已解析的文档, 按新参数重新切一遍, 立刻返回结果.
不落库、不做向量化, 所以是毫秒级的. 满意了再点"应用并重新处理"。

解析结果缓存
------------
预览的前提是不重复解析 PDF(解析要 1~2 秒, 每次调参都解析会很难用).
所以把 CleanDocument 缓存在内存里, 按 (doc_id, file_md5) 做键 ——
文件内容一变 hash 就变, 缓存自动失效, 不需要额外的一致性维护.

缓存容量刻意设得很小(默认 4 份): 一份 200 页文档的 CleanDocument 可能有几十 MB,
无上限缓存会把内存吃光. 这是个明确的"用 CPU 换内存"的取舍.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError, ParamInvalidError
from app.core.logging import get_logger, log_kv
from app.models.document import Chunk as ChunkModel
from app.models.document import ChunkType
from app.services.chunking import ChunkParams, build_embedding_text, chunk_document, summarize
from app.services.parser import CleanDocument, clean_document, parse_pdf

logger = get_logger("docmind.chunk_preview")

#: 解析结果缓存容量. 见模块文档里关于"用 CPU 换内存"的说明.
_CACHE_LIMIT = 4

_cache: OrderedDict[str, CleanDocument] = OrderedDict()
_cache_lock = threading.Lock()


def get_clean_document(doc_id: str, file_path: str, file_md5: str) -> CleanDocument:
    """获取(或构建)文档的清洗结果, 带缓存."""
    key = f"{doc_id}:{file_md5}"

    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached

    # 解析放到锁外 —— 它要 1~2 秒, 占着锁会让所有预览请求排队
    parsed = parse_pdf(file_path)
    cleaned = clean_document(parsed)

    with _cache_lock:
        _cache[key] = cleaned
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_LIMIT:
            _cache.popitem(last=False)

    log_kv(logger, "chunk_preview.parsed", doc_id=doc_id, chars=cleaned.char_count)
    return cleaned


def invalidate(doc_id: str) -> None:
    """文档被删除或重新处理时清掉它的解析缓存."""
    with _cache_lock:
        for key in [k for k in _cache if k.startswith(f"{doc_id}:")]:
            _cache.pop(key, None)


def preview(
    clean: CleanDocument,
    doc_id: str,
    params: ChunkParams,
    *,
    limit: int = 200,
) -> dict[str, Any]:
    """按给定参数重新切分, 返回统计与分块明细(不落库)."""
    params.validate()

    result = chunk_document(clean, doc_id, params=params)
    stats = summarize(result, params)

    parents = sorted(result.parents, key=lambda c: c.order_index)
    children = sorted(result.children, key=lambda c: c.order_index)
    parent_by_id = {p.id: p for p in parents}

    # 父块带上自己的子块, 让前端能直接渲染成两级结构.
    # 只截取前 limit 个, 避免超大文档把响应撑爆.
    parent_items = []
    for parent in parents[:limit]:
        parent_items.append(
            {
                "id": parent.id,
                "type": ChunkType.PARENT.value,
                "content": parent.content,
                "char_count": parent.char_count,
                "page_start": parent.page_start,
                "page_end": parent.page_end,
                "section_path": parent.section_path,
                "order_index": parent.order_index,
                "child_ids": [c.id for c in children if c.parent_id == parent.id],
            }
        )

    child_items = []
    for child in children[:limit]:
        parent = parent_by_id.get(child.parent_id or "")
        child_items.append(
            {
                "id": child.id,
                "type": ChunkType.CHILD.value,
                "parent_id": child.parent_id,
                "content": child.content,
                "embedding_text": child.embedding_text,
                "char_count": child.char_count,
                "page_start": child.page_start,
                "page_end": child.page_end,
                "section_path": child.section_path,
                "order_index": child.order_index,
                "parent_char_count": parent.char_count if parent else 0,
                # 前端用它标注"送去算向量的文本与原文不同"
                "embedding_differs": child.embedding_text != child.content,
            }
        )

    return {
        "stats": stats,
        "parents": parent_items,
        "children": child_items,
        "truncated": len(children) > limit or len(parents) > limit,
    }


async def list_stored_chunks(
    session: AsyncSession,
    doc_id: str,
    *,
    limit: int = 300,
    include_parents: bool = True,
) -> dict[str, Any]:
    """读取**已落库**的分块(即当前实际生效的切分结果).

    与 preview 的区别: preview 是"按新参数试切", 这里读的是"当初入库时真正用的结果".
    两者并排展示, 用户才能看出调参到底改变了什么.
    """
    stmt = select(ChunkModel).where(ChunkModel.doc_id == doc_id).order_by(ChunkModel.order_index)
    if not include_parents:
        stmt = stmt.where(ChunkModel.chunk_type == ChunkType.CHILD.value)

    rows = list((await session.execute(stmt)).scalars().all())
    if not rows and include_parents:
        # 分块表为空但文档存在时, 说明那次入库没成功 —— 交给上层判断, 这里不报错
        pass

    parents = [r for r in rows if r.chunk_type == ChunkType.PARENT.value]
    children = [r for r in rows if r.chunk_type == ChunkType.CHILD.value]
    parent_size = {p.id: p.char_count for p in parents}

    def serialize(row: ChunkModel) -> dict[str, Any]:
        is_child = row.chunk_type == ChunkType.CHILD.value
        content = row.content
        # 必须复用入库时用的同一份逻辑, 否则界面上显示的"送入向量的文本"
        # 与实际入库时用的不一致 —— 用户会基于错误信息做调参决策.
        embedding_text = build_embedding_text(content, row.section_path)
        return {
            "id": row.id,
            "type": row.chunk_type,
            "parent_id": row.parent_id,
            "content": content,
            "embedding_text": embedding_text,
            "char_count": row.char_count,
            "page_start": row.page_start,
            "page_end": row.page_end,
            "section_path": row.section_path,
            "order_index": row.order_index,
            "is_indexed": row.is_indexed,
            "parent_char_count": parent_size.get(row.parent_id or "", 0) if is_child else 0,
            "embedding_differs": is_child and embedding_text != content,
        }

    return {
        "doc_id": doc_id,
        "total": len(rows),
        "parents": [serialize(r) for r in parents[:limit]],
        "children": [serialize(r) for r in children[:limit]],
        "truncated": len(rows) > limit,
    }


def ensure_document_exists(document: Any) -> None:
    if document is None:
        raise NotFoundError("文档不存在")


def validate_params_or_raise(raw: dict[str, Any]) -> ChunkParams:
    """把请求体转成参数对象; 非法参数直接抛 400."""
    try:
        return ChunkParams.from_dict(raw)
    except ParamInvalidError:
        raise
    except (TypeError, ValueError) as exc:
        raise ParamInvalidError(f"分块参数不合法: {exc}") from exc
