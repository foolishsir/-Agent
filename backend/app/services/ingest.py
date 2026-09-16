"""文档入库编排: 解析 → 清洗 → 分块 → 向量化 → 写入向量库.

为什么要把 CPU 密集步骤放进 ``asyncio.to_thread``
------------------------------------------------
FastAPI 是单事件循环模型. 在协程里直接调用一个跑 3 秒的纯 CPU 函数,
**整个进程在这 3 秒内无法处理任何其他请求** —— 包括健康检查.

``to_thread`` 把工作挪到线程池. 需要说明的是: 纯 Python 代码受 GIL 限制,
并不会真正并行; 但 numpy / torch 在做矩阵运算时会主动释放 GIL,
所以 embedding 这一步是**真并行**的, 而解析/清洗至少能保证事件循环不被阻塞.

幂等性设计
----------
整个流程可以安全重跑, 靠三点保证:
1. chunk id 确定性生成(``doc_id`` + 序号), 与处理时间无关
2. 写入前先按 ``doc_id`` 清空向量库中的旧数据
3. DB 里先删旧 chunks 再插新的

所以"失败重试"不会产生重复数据, 只会覆盖. 这一点很重要 ——
脏数据比没数据更难排查.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import AppException, DocumentParseError
from app.core.logging import get_logger, log_kv
from app.db.session import get_session_factory
from app.models.document import Chunk as ChunkModel
from app.models.document import ChunkType, Document, DocumentStatus
from app.services.chunking import Chunk, chunk_document
from app.services.embedding import get_embedding_provider
from app.services.parser import clean_document, parse_pdf
from app.services.vectorstore import SearchFilter, VectorRecord, get_vector_store

logger = get_logger("docmind.ingest")


@dataclass
class IngestResult:
    doc_id: str
    status: DocumentStatus
    page_count: int
    parent_chunks: int
    child_chunks: int
    char_count: int
    parse_cost_ms: int
    embed_cost_ms: int
    error: str | None = None


async def submit_ingest(doc_id: str) -> IngestResult:
    """投递(并执行)入库任务.

    ``inline`` 模式下同步等待完成 —— 开发环境不需要额外装 Redis,
    "clone 下来就能跑" 对开源项目的首次体验很重要.
    ``queue`` 模式下由 RQ Worker 消费(P4 阶段接入), 这里会立刻返回.

    两种模式下**任务函数本身完全相同**, 因为任务定义只是 service 的薄包装.
    """
    if settings.task_mode == "inline":
        return await run_ingest(doc_id)

    # P4 阶段实现: 投递到 RQ 队列并立即返回 PENDING
    from app.worker.queue import enqueue_ingest  # noqa: PLC0415 - 避免循环导入

    await asyncio.to_thread(enqueue_ingest, doc_id)
    return IngestResult(
        doc_id=doc_id,
        status=DocumentStatus.PENDING,
        page_count=0,
        parent_chunks=0,
        child_chunks=0,
        char_count=0,
        parse_cost_ms=0,
        embed_cost_ms=0,
    )


async def run_ingest(doc_id: str) -> IngestResult:
    """执行完整的文档入库流程."""
    total_started = time.perf_counter()
    parse_cost_ms = 0
    embed_cost_ms = 0

    factory = get_session_factory()

    # ---------------- 1. 读取文档并进入 PARSING 状态 ----------------
    async with factory() as session:
        document = await session.get(Document, doc_id)
        if document is None:
            raise DocumentParseError(f"文档不存在: {doc_id}")

        if document.status == DocumentStatus.READY.value:
            logger.info("文档已就绪, 跳过重复处理 | doc_id=%s", doc_id)
            return _result_of(document)

        document.status = DocumentStatus.PARSING.value
        document.error_msg = None
        await session.commit()

        file_path = document.file_path
        filename = document.filename
        user_id = document.user_id

    log_kv(logger, "ingest.start", doc_id=doc_id, file=filename, user_id=user_id)

    try:
        # ---------------- 2. 解析(PDF 坐标 / 双栏 / 页眉页脚) ----------------
        parse_started = time.perf_counter()
        parsed = await asyncio.to_thread(parse_pdf, file_path, filename=filename)
        parse_cost_ms = int((time.perf_counter() - parse_started) * 1000)

        if parsed.is_scanned:
            raise DocumentParseError(
                f"检测到扫描件(共 {parsed.page_count} 页, 几乎没有文本层). "
                "当前版本未内置 OCR, 请上传电子版 PDF, "
                "或安装 OCR 依赖后重新处理."
            )

        if parsed.char_count == 0:
            raise DocumentParseError("未能从文档中提取到任何文本内容")

        # ---------------- 3. 清洗 ----------------
        cleaned = await asyncio.to_thread(clean_document, parsed)
        if not cleaned.paragraphs:
            raise DocumentParseError("文档清洗后没有可用内容")

        # ---------------- 4. 父子块切分 ----------------
        chunking = await asyncio.to_thread(chunk_document, cleaned, doc_id)

        # ---------------- 5. 向量化(状态切到 EMBEDDING) ----------------
        async with factory() as session:
            document = await session.get(Document, doc_id)
            if document is None:
                raise DocumentParseError(f"文档在入库过程中被删除: {doc_id}")
            document.status = DocumentStatus.EMBEDDING.value
            await session.commit()

        embed_started = time.perf_counter()
        records = await _embed_children(
            chunking.children,
            user_id=user_id,
            filename=filename,
            doc_id=doc_id,
        )
        embed_cost_ms = int((time.perf_counter() - embed_started) * 1000)

        # ---------------- 6. 写入向量库 ----------------
        # 先删后写保证幂等: 重跑时不会留下上一轮的遗留分块.
        # 这一步也顺带处理了"上次处理写到一半失败了"的残留.
        store = get_vector_store()
        await asyncio.to_thread(store.delete_by_doc, doc_id)
        indexed = await asyncio.to_thread(store.upsert, records)

        # ---------------- 7. 落库分块与最终状态 ----------------
        async with factory() as session:
            document = await session.get(Document, doc_id)
            if document is None:
                raise DocumentParseError(f"文档在入库过程中被删除: {doc_id}")

            await _replace_chunks(session, doc_id, chunking.parents, chunking.children)

            document.status = DocumentStatus.READY.value
            document.page_count = cleaned.page_count
            document.char_count = cleaned.char_count
            document.parent_chunk_count = len(chunking.parents)
            document.child_chunk_count = len(chunking.children)
            document.chunks_indexed = indexed
            document.parse_cost_ms = parse_cost_ms
            document.embed_cost_ms = embed_cost_ms
            document.error_msg = None
            await session.commit()

        # 文档内容变了 → BM25 索引缓存必须失效.
        # 不失效的话关键词检索会继续用旧语料, 表现为"新上传的文档搜不到",
        # 更危险的是"已删除的文档还能搜到" —— 那是数据泄露.
        _invalidate_bm25_cache()

        total_ms = int((time.perf_counter() - total_started) * 1000)
        log_kv(
            logger,
            "ingest.done",
            doc_id=doc_id,
            pages=cleaned.page_count,
            parents=len(chunking.parents),
            children=len(chunking.children),
            chars=cleaned.char_count,
            parse_ms=parse_cost_ms,
            embed_ms=embed_cost_ms,
            total_ms=total_ms,
        )

        return IngestResult(
            doc_id=doc_id,
            status=DocumentStatus.READY,
            page_count=cleaned.page_count,
            parent_chunks=len(chunking.parents),
            child_chunks=len(chunking.children),
            char_count=cleaned.char_count,
            parse_cost_ms=parse_cost_ms,
            embed_cost_ms=embed_cost_ms,
        )

    except Exception as exc:  # noqa: BLE001 - 任何失败都要落到 FAILED 状态
        message = exc.message if isinstance(exc, AppException) else str(exc)
        logger.exception("文档入库失败 | doc_id=%s error=%s", doc_id, message)
        await _mark_failed(doc_id, message)
        return IngestResult(
            doc_id=doc_id,
            status=DocumentStatus.FAILED,
            page_count=0,
            parent_chunks=0,
            child_chunks=0,
            char_count=0,
            parse_cost_ms=parse_cost_ms,
            embed_cost_ms=embed_cost_ms,
            error=message,
        )


# --------------------------------------------------------------------------- #
# 内部步骤
# --------------------------------------------------------------------------- #
async def _embed_children(
    children: list[Chunk], *, user_id: str, filename: str, doc_id: str
) -> list[VectorRecord]:
    """把子块编码成向量记录.

    注意送入模型的是 ``chunk.embedding_text`` 而不是 ``chunk.content`` ——
    前者在正文前拼了章节路径以补足语境, 后者是存入向量库供展示与关键词检索的干净原文.
    两者的分工见 ``Chunk.embedding_text`` 的注释.
    """
    if not children:
        return []

    provider = get_embedding_provider()
    texts = [chunk.embedding_text for chunk in children]
    vectors = await asyncio.to_thread(provider.encode_passages, texts)

    if len(vectors) != len(children):
        raise DocumentParseError(
            f"向量数量({len(vectors)})与分块数量({len(children)})不一致, 拒绝写入"
        )

    return [
        VectorRecord(
            id=chunk.id,
            embedding=vector,
            document=chunk.content,
            metadata={
                "doc_id": doc_id,
                "parent_id": chunk.parent_id,
                "chunk_type": ChunkType.CHILD.value,
                "user_id": user_id,
                "filename": filename,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_path": chunk.section_path or "",
                "order_index": chunk.order_index,
            },
        )
        for chunk, vector in zip(children, vectors, strict=True)
    ]


async def _replace_chunks(
    session: AsyncSession,
    doc_id: str,
    parents: list[Chunk],
    children: list[Chunk],
) -> None:
    """先删旧分块再插入新分块, 保证重跑不产生重复."""
    await session.execute(delete(ChunkModel).where(ChunkModel.doc_id == doc_id))

    session.add_all(
        [
            ChunkModel(
                id=chunk.id,
                doc_id=chunk.doc_id,
                parent_id=chunk.parent_id,
                chunk_type=chunk.chunk_type.value,
                content=chunk.content,
                char_count=chunk.char_count,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_path=chunk.section_path,
                order_index=chunk.order_index,
                is_indexed=chunk.chunk_type == ChunkType.CHILD,
            )
            for chunk in (*parents, *children)
        ]
    )
    await session.flush()


def _invalidate_bm25_cache() -> None:
    """让关键词检索的语料缓存失效.

    放在 ingest 和 delete 两处调用. 漏掉任何一处都会导致
    "检索结果与实际文档不一致" —— 这类 bug 不会报错, 只会让答案莫名其妙.
    另外 BM25 缓存本身还有 TTL 兜底, 但那是最后一道防线, 不是借口.
    """
    try:
        from app.services.retrieval.bm25 import get_bm25_retriever  # noqa: PLC0415

        get_bm25_retriever().invalidate()
    except Exception:  # noqa: BLE001 - 缓存失效失败不应影响入库结果
        logger.exception("BM25 缓存失效失败")


async def _mark_failed(doc_id: str, message: str) -> None:
    """把文档标记为失败并记录原因."""
    factory = get_session_factory()
    try:
        async with factory() as session:
            document = await session.get(Document, doc_id)
            if document is None:
                return
            document.status = DocumentStatus.FAILED.value
            # 错误信息可能很长(比如 PyMuPDF 的完整异常), 截断避免撑爆字段
            document.error_msg = message[:2000]
            await session.commit()
    except Exception:  # noqa: BLE001 - 标记失败本身不能掩盖原始异常
        logger.exception("标记文档失败状态时出错 | doc_id=%s", doc_id)


def _result_of(document: Document) -> IngestResult:
    return IngestResult(
        doc_id=document.id,
        status=DocumentStatus(document.status),
        page_count=document.page_count,
        parent_chunks=document.parent_chunk_count,
        child_chunks=document.child_chunk_count,
        char_count=document.char_count,
        parse_cost_ms=document.parse_cost_ms,
        embed_cost_ms=document.embed_cost_ms,
    )


async def count_searchable_chunks(user_id: str) -> int:
    """统计某用户可检索的分块总数(用于就绪探针与调试)."""
    store = get_vector_store()
    return await asyncio.to_thread(store.count, SearchFilter(user_id=user_id))


async def list_chunks_from_db(session: AsyncSession, doc_id: str) -> list[ChunkModel]:
    """从关系库读取某文档的全部分块(BM25 语料的备用来源)."""
    result = await session.execute(
        select(ChunkModel).where(ChunkModel.doc_id == doc_id).order_by(ChunkModel.order_index)
    )
    return list(result.scalars().all())
