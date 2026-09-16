"""文档管理服务: 上传 / 查询 / 删除.

架构约定
--------
本模块**不依赖 FastAPI** —— 它接收的是 ``AsyncIterator[bytes]`` 而不是 ``UploadFile``,
由 API 层做适配. 好处是同一份逻辑可以直接被异步 Worker、CLI 脚本、
甚至消息队列消费者复用.

三个必须处理好的工程问题
------------------------
1. **幂等上传**: 用文件内容 MD5 作为幂等键. 用户网络卡顿点了两次上传,
   不应该在向量库里存两份数据.
2. **不信任客户端**: 扩展名可以随便改, 必须校验文件头魔数.
3. **删除的一致性**: 关系库与向量库没有跨库事务, 采用"先软删标记 → 再清向量"
   的顺序, 保证"宁可短暂不可见, 也不能检索到已删内容".
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import aiofiles
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    ConflictError,
    FileTooLargeError,
    NotFoundError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger, log_kv
from app.models.document import Chunk, Document, DocumentStatus

logger = get_logger("docmind.documents")

#: PDF 文件头魔数. 只看扩展名是不够的 —— 把 .zip 改名成 .pdf 就能绕过.
_PDF_MAGIC = b"%PDF"

_READ_CHUNK = 1024 * 1024  # 1MB


async def create_document(
    session: AsyncSession,
    *,
    filename: str,
    stream: AsyncIterator[bytes],
    user_id: str,
) -> tuple[Document, bool]:
    """保存上传的文件并创建文档记录.

    Returns:
        ``(document, created)``. ``created=False`` 表示命中幂等, 复用了已有文档.

    Raises:
        UnsupportedFileTypeError: 扩展名或文件头不合法
        FileTooLargeError: 超过大小限制
        ConflictError: 同名内容正在处理中
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise UnsupportedFileTypeError(
            f"不支持的文件类型 {suffix!r}, 当前仅支持: {', '.join(settings.allowed_extensions)}"
        )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    tmp_path = settings.upload_dir / f".upload_{uuid.uuid4().hex}"
    md5, size = await _write_stream_to_temp(stream, tmp_path, max_bytes)

    try:
        await _assert_is_pdf(tmp_path)
    except Exception:
        _safe_unlink(tmp_path)
        raise

    # 文件名用内容 MD5 而不是原始文件名:
    # 中文文件名、重名文件、路径穿越("../")等问题一次性全部规避
    final_path = settings.upload_dir / f"{md5}{suffix}"

    # ---------------- 幂等检查 ----------------
    existing = await _find_by_md5(session, user_id, md5)
    if existing is not None:
        if existing.status == DocumentStatus.DELETED.value:
            # 曾经删过同一份文件 → 视为重新上传, 复用记录并重新走入库流程.
            # 注意: 删除时原文件已被清掉, 所以这里必须把临时文件搬到最终路径,
            # 否则后续解析会读到不存在的文件而失败.
            await asyncio.to_thread(os.replace, tmp_path, final_path)
            await _reuse_deleted(session, existing, final_path, filename, size)
            log_kv(logger, "upload.reuse_deleted", doc_id=existing.id, md5=md5)
            return existing, True

        if existing.status == DocumentStatus.FAILED.value:
            # 上次处理失败 → 复用记录并重新走一遍流程
            if not final_path.exists():
                await asyncio.to_thread(os.replace, tmp_path, final_path)
            else:
                _safe_unlink(tmp_path)
            existing.status = DocumentStatus.PENDING.value
            existing.error_msg = None
            existing.file_path = str(final_path)
            await session.commit()
            log_kv(logger, "upload.retry_failed", doc_id=existing.id, md5=md5)
            return existing, True

        if existing.is_processing:
            _safe_unlink(tmp_path)
            log_kv(logger, "upload.duplicate_processing", doc_id=existing.id, md5=md5)
            return existing, False

        # 已经 READY → 秒传
        _safe_unlink(tmp_path)
        log_kv(logger, "upload.duplicate_ready", doc_id=existing.id, md5=md5)
        return existing, False

    # ---------------- 新文档 ----------------
    await asyncio.to_thread(os.replace, tmp_path, final_path)

    document = Document(
        user_id=user_id,
        filename=filename,
        file_md5=md5,
        file_path=str(final_path),
        file_size=size,
        status=DocumentStatus.PENDING.value,
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)

    log_kv(
        logger,
        "upload.saved",
        doc_id=document.id,
        filename=filename,
        size_kb=round(size / 1024, 1),
        md5=md5,
    )
    return document, True


async def _write_stream_to_temp(
    stream: AsyncIterator[bytes], tmp_path: Path, max_bytes: int
) -> tuple[str, int]:
    """边落盘边算 MD5, 并强制大小上限.

    为什么不先全部读进内存: 50MB 的 PDF 读进内存后, 并发 10 个上传就是 500MB.
    流式处理的内存占用是常数级的, 与文件大小无关.
    """
    hasher = hashlib.md5()  # noqa: S324 - 用于内容去重, 非安全用途
    size = 0

    try:
        async with aiofiles.open(tmp_path, "wb") as handle:
            async for block in stream:
                if not block:
                    continue
                size += len(block)
                if size > max_bytes:
                    raise FileTooLargeError(f"文件超过 {settings.max_upload_size_mb}MB 上限")
                hasher.update(block)
                await handle.write(block)
    except Exception:
        # 任何失败都要清理临时文件, 否则上传目录会被残片填满
        _safe_unlink(tmp_path)
        raise

    if size == 0:
        _safe_unlink(tmp_path)
        raise UnsupportedFileTypeError("上传的文件内容为空")

    return hasher.hexdigest(), size


async def _assert_is_pdf(path: Path) -> None:
    """校验文件头魔数, 而不是只信扩展名."""
    async with aiofiles.open(path, "rb") as handle:
        head = await handle.read(len(_PDF_MAGIC))
    if not head.startswith(_PDF_MAGIC):
        raise UnsupportedFileTypeError("文件内容不是有效的 PDF(文件头校验失败)")


async def _find_by_md5(session: AsyncSession, user_id: str, md5: str) -> Document | None:
    result = await session.execute(
        select(Document).where(Document.user_id == user_id, Document.file_md5 == md5)
    )
    return result.scalars().first()


async def _reuse_deleted(
    session: AsyncSession, document: Document, final_path: Path, filename: str, size: int
) -> None:
    """复用一条曾被软删的记录."""
    document.status = DocumentStatus.PENDING.value
    document.error_msg = None
    document.filename = filename
    document.file_path = str(final_path)
    document.file_size = size
    document.page_count = 0
    document.char_count = 0
    document.parent_chunk_count = 0
    document.child_chunk_count = 0
    document.chunks_indexed = 0
    await session.commit()


# --------------------------------------------------------------------------- #
# 查询
# --------------------------------------------------------------------------- #
async def list_documents(
    session: AsyncSession,
    *,
    user_id: str,
    page: int = 1,
    page_size: int = 20,
    status: str | None = None,
) -> tuple[int, list[Document]]:
    """分页列出文档(不含已删除的)."""
    conditions = [
        Document.user_id == user_id,
        Document.status != DocumentStatus.DELETED.value,
    ]
    if status:
        conditions.append(Document.status == status)

    total = int(
        (
            await session.execute(select(func.count()).select_from(Document).where(*conditions))
        ).scalar_one()
    )

    result = await session.execute(
        select(Document)
        .where(*conditions)
        .order_by(Document.created_at.desc())
        .offset(max(0, (page - 1) * page_size))
        .limit(page_size)
    )
    return total, list(result.scalars().all())


async def get_document(
    session: AsyncSession, doc_id: str, *, user_id: str | None = None
) -> Document:
    """按 id 取文档; 不存在或不属于该用户时抛 404.

    注意: **不区分"不存在"和"不属于你"** —— 都返回 404.
    如果对不属于自己的资源返回 403, 攻击者就能通过状态码差异
    枚举出系统里有哪些 doc_id. 这是"越权探测"的经典防线.
    """
    document = await session.get(Document, doc_id)
    if document is None or document.status == DocumentStatus.DELETED.value:
        raise NotFoundError(f"文档不存在: {doc_id}")
    if user_id is not None and document.user_id != user_id:
        raise NotFoundError(f"文档不存在: {doc_id}")
    return document


# --------------------------------------------------------------------------- #
# 删除
# --------------------------------------------------------------------------- #
async def delete_document(
    session: AsyncSession,
    doc_id: str,
    *,
    user_id: str | None = None,
    purge_file: bool = True,
) -> dict[str, int | bool]:
    """删除文档.

    三段式删除, 顺序是刻意设计的::

        ① 关系库标记 DELETED      → 立即对用户不可见(秒级生效)
        ② 清理向量库中的分块       → 可能失败, 失败也不影响 ①
        ③ 删除原文件(可选)

    为什么不是"先删向量再改状态": 如果先删向量时进程崩了, 用户看到文档还在,
    但问答已经检索不到内容 —— 状态与事实不一致. 反过来则是"状态说没了,
    底层残留一点数据", 用户感知一致, 残留在重试或后续清理中收敛.

    兜底: 检索时会用 ``doc_id in (有效文档)`` 过滤, 即使清理失败也检索不到已删内容.
    """
    document = await get_document(session, doc_id, user_id=user_id)

    # ① 立即软删
    document.status = DocumentStatus.DELETED.value
    document.chunks_indexed = 0
    await session.commit()

    # ② 清理关系库分块
    deleted_chunks = (
        await session.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
    ).rowcount or 0
    await session.commit()

    # ③ 清理向量库(独立存储, 失败不回滚 ①②)
    deleted_vectors = 0
    try:
        from app.services.vectorstore import get_vector_store  # noqa: PLC0415

        store = get_vector_store()
        deleted_vectors = await asyncio.to_thread(store.delete_by_doc, doc_id)
    except Exception:  # noqa: BLE001
        logger.exception("向量清理失败, 已保留软删标记供后续重试 | doc_id=%s", doc_id)

    # ④ 删除原文件
    file_removed = False
    if purge_file:
        try:
            path = Path(document.file_path)
            if path.exists():
                await asyncio.to_thread(path.unlink)
                file_removed = True
        except OSError:
            logger.warning("删除原文件失败 | doc_id=%s path=%s", doc_id, document.file_path)

    log_kv(
        logger,
        "document.deleted",
        doc_id=doc_id,
        chunks=deleted_chunks,
        vectors=deleted_vectors,
        file_removed=file_removed,
    )

    # 文档删了但 BM25 语料缓存还留着旧内容 → 会检索到已删除文档的片段.
    # 这是"幽灵数据"最隐蔽的表现形式: 存储里查不到, 但检索能召回.
    try:
        from app.services.retrieval.bm25 import get_bm25_retriever  # noqa: PLC0415

        get_bm25_retriever().invalidate()
    except Exception:  # noqa: BLE001 - 缓存失效失败不应影响删除结果
        logger.exception("BM25 缓存失效失败 | doc_id=%s", doc_id)

    return {
        "doc_id": doc_id,
        "deleted_chunks": deleted_chunks,
        "deleted_vectors": deleted_vectors,
        "file_removed": file_removed,
    }


async def ensure_ready(document: Document) -> None:
    """确认文档已完成向量化, 否则抛冲突错误.

    问答接口在检索前必须调用 —— 否则用户会在文档还在解析时提问,
    得到"文档里没有相关内容"的误导性回答.
    """
    if document.status == DocumentStatus.READY.value:
        return
    if document.status == DocumentStatus.FAILED.value:
        raise ConflictError(f"文档处理失败, 无法问答: {document.error_msg or '未知原因'}")
    raise ConflictError(f"文档尚未完成处理(当前状态: {document.status}), 请稍后再试")


def _safe_unlink(path: Path) -> None:
    """删除临时文件, 失败不影响主流程."""
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover
        logger.warning("清理临时文件失败 | path=%s", path)
