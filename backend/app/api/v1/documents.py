"""文档管理接口.

接口清单
--------
============================  ======  ==========================================
方法                           路径     说明
============================  ======  ==========================================
POST                          /       上传 PDF(幂等, 内容 MD5 去重)
GET                           /       分页列表(支持状态筛选)
GET                           /{id}   文档详情
GET                           /{id}/status  精简状态(供前端高频轮询)
POST                          /{id}/reindex 重新处理(修复失败或换分块参数后重跑)
DELETE                        /{id}   删除(软删 + 清理向量 + 删文件)
============================  ======  ==========================================
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, File, Query, UploadFile, status

from app.api.deps import CurrentUser, SessionDep
from app.core.response import PageData, ok
from app.schemas.document import (
    DeleteResponse,
    DocumentOut,
    IngestStatsOut,
    UploadResponse,
    build_status_out,
)
from app.services import document_service
from app.services.ingest import submit_ingest

router = APIRouter()

#: 流式读取上传文件的分片大小
_UPLOAD_CHUNK = 1024 * 1024


async def _iter_upload(file: UploadFile) -> AsyncIterator[bytes]:
    """把 FastAPI 的 UploadFile 适配成分块字节流.

    由 API 层做这层适配, service 层才能保持"不依赖 Web 框架".
    服务层接收 ``AsyncIterator[bytes]``, 因此同一个 service 函数
    可以被 Worker、CLI 脚本、消息队列消费者直接复用.
    """
    while True:
        block = await file.read(_UPLOAD_CHUNK)
        if not block:
            break
        yield block


@router.post(
    "",
    summary="上传文档",
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    session: SessionDep,
    user_id: CurrentUser,
    file: Annotated[UploadFile, File(description="PDF 文件, 默认上限 50MB")],
) -> dict[str, Any]:
    """上传 PDF 并触发解析入库.

    **幂等**: 以文件内容 MD5 为幂等键. 同一份文件重复上传会直接复用已有文档
    (``created=false``), 不会重复解析, 也不会在向量库里产生重复数据.
    """
    document, created = await document_service.create_document(
        session,
        filename=file.filename or "unnamed.pdf",
        stream=_iter_upload(file),
        user_id=user_id,
    )

    if created:
        # inline 模式下这里会同步跑完整个入库流程; queue 模式下立即返回,
        # 前端通过 /{id}/status 轮询进度.
        await submit_ingest(document.id)
        # 入库过程在另一个会话里改了状态, 这里必须刷新才能拿到最新值
        await session.refresh(document)

    payload = UploadResponse(
        document=DocumentOut.model_validate(document),
        created=created,
        message="上传成功, 已完成解析入库" if created else "该文件已存在, 直接复用",
    )
    return ok(payload.model_dump(mode="json"))


@router.get("", summary="文档列表")
async def list_documents(
    session: SessionDep,
    user_id: CurrentUser,
    page: Annotated[int, Query(ge=1, description="页码, 从 1 开始")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="每页条数")] = 20,
    doc_status: Annotated[
        str | None,
        Query(alias="status", description="按状态筛选: PENDING/PARSING/EMBEDDING/READY/FAILED"),
    ] = None,
) -> dict[str, Any]:
    """分页列出当前用户的文档(不含已删除)."""
    total, items = await document_service.list_documents(
        session, user_id=user_id, page=page, page_size=page_size, status=doc_status
    )
    page_data = PageData[DocumentOut](
        total=total,
        page=page,
        page_size=page_size,
        items=[DocumentOut.model_validate(item) for item in items],
    )
    return ok(page_data.model_dump(mode="json"))


@router.get("/{doc_id}", summary="文档详情")
async def get_document(
    session: SessionDep,
    user_id: CurrentUser,
    doc_id: str,
) -> dict[str, Any]:
    """查看单个文档的处理结果与统计."""
    document = await document_service.get_document(session, doc_id, user_id=user_id)
    return ok(DocumentOut.model_validate(document).model_dump(mode="json"))


@router.get("/{doc_id}/status", summary="文档处理状态(轮询)")
async def get_document_status(
    session: SessionDep,
    user_id: CurrentUser,
    doc_id: str,
) -> dict[str, Any]:
    """精简状态查询, 供前端轮询.

    单独开一个接口而不是让前端反复拉详情, 是因为轮询频率高(每秒级),
    响应体越小越好.
    """
    document = await document_service.get_document(session, doc_id, user_id=user_id)
    return ok(build_status_out(document).model_dump(mode="json"))


@router.post("/{doc_id}/reindex", summary="重新处理文档")
async def reindex_document(
    session: SessionDep,
    user_id: CurrentUser,
    doc_id: str,
) -> dict[str, Any]:
    """重新跑一遍完整入库流程.

    典型用途: 处理失败后重试; 调整了分块参数后想让效果生效.
    因为 chunk id 是确定性生成的, 重跑是**覆盖**而不是追加, 不会产生重复数据.
    """
    document = await document_service.get_document(session, doc_id, user_id=user_id)

    from app.models.document import DocumentStatus  # noqa: PLC0415

    document.status = DocumentStatus.PENDING.value
    document.error_msg = None
    await session.commit()

    result = await submit_ingest(doc_id)
    return ok(IngestStatsOut(**vars(result)).model_dump(mode="json"))


@router.delete("/{doc_id}", summary="删除文档")
async def delete_document(
    session: SessionDep,
    user_id: CurrentUser,
    doc_id: str,
) -> dict[str, Any]:
    """删除文档: 软删标记 → 清理关系库分块 → 清理向量 → 删除原文件.

    顺序是刻意设计的, 详见 ``document_service.delete_document`` 的文档字符串.
    """
    result = await document_service.delete_document(session, doc_id, user_id=user_id)
    return ok(DeleteResponse(**result).model_dump(mode="json"))  # type: ignore[arg-type]
