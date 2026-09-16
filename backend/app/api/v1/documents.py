"""文档管理接口.

接口清单
--------
============================  ======  ==========================================
方法                           路径     说明
============================  ======  ==========================================
POST                          /       上传 PDF(幂等, 内容 MD5 去重)
GET                           /       分页列表(支持状态筛选)
GET                           /chunk-strategies  可选分块策略与参数元数据
GET                           /{id}   文档详情
GET                           /{id}/status  精简状态(供前端高频轮询)
GET                           /{id}/chunks  查看**已落库**的分块内容
POST                          /{id}/chunk-preview  按新参数试切, 不落库
POST                          /{id}/chunk-apply    保存参数并重新处理
POST                          /{id}/reindex 重新处理(修复失败后重跑)
DELETE                        /{id}   删除(软删 + 清理向量 + 删文件)
============================  ======  ==========================================
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Body, File, Query, UploadFile, status

from app.api.deps import CurrentUser, SessionDep
from app.core.response import PageData, ok
from app.schemas.document import (
    DeleteResponse,
    DocumentOut,
    IngestStatsOut,
    UploadResponse,
    build_status_out,
)
from app.services import chunk_preview, document_service
from app.services.chunking import AVAILABLE_STRATEGIES, STRATEGY_LABELS, ChunkParams
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


# --------------------------------------------------------------------------- #
# 分块查看与调参
# --------------------------------------------------------------------------- #
# 注意: 这条路由必须注册在 "/{doc_id}" **之前**.
# FastAPI 按注册顺序匹配, 而 "/chunk-strategies" 完全符合 "/{doc_id}" 的形状 ——
# 顺序反了的话, 请求会被当成"查询 id 为 chunk-strategies 的文档"并返回 404.
# 这类"静态路径被动态路径吃掉"的问题很常见, 修法就是把静态路径放前面.
@router.get("/chunk-strategies", summary="可选分块策略与参数元数据")
async def chunk_strategies() -> dict[str, Any]:
    """返回可选策略与当前生效的参数, 供前端渲染调参面板.

    元数据由后端下发而不是前端硬编码: 新增一种策略只需要改后端, 前端一行都不用动.
    """
    current = ChunkParams.from_settings()
    return ok(
        {
            "strategies": [
                {"value": name, "label": STRATEGY_LABELS.get(name, name)}
                for name in AVAILABLE_STRATEGIES
            ],
            "current": current.to_dict(),
            "limits": {
                "parent_size": {"min": 100, "max": 4000, "step": 50},
                "child_size": {"min": 20, "max": 1200, "step": 10},
                "overlap": {"min": 0, "max": 300, "step": 10},
                "min_size": {"min": 0, "max": 200, "step": 5},
            },
        }
    )


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
    chunk_preview.invalidate(doc_id)
    return ok(DeleteResponse(**result).model_dump(mode="json"))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 分块查看与调参
# --------------------------------------------------------------------------- #
@router.get("/{doc_id}/chunks", summary="查看已落库的分块")
async def list_chunks(
    session: SessionDep,
    user_id: CurrentUser,
    doc_id: str,
    limit: Annotated[int, Query(ge=1, le=1000, description="最多返回多少条")] = 300,
) -> dict[str, Any]:
    """查看这份文档**当初入库时实际使用**的分块结果.

    与 ``chunk-preview`` 的区别: 这里读的是真实落库的数据,
    预览接口是按新参数试切. 两者并排看, 才知道调参到底改变了什么.
    """
    await document_service.get_document(session, doc_id, user_id=user_id)
    return ok(await chunk_preview.list_stored_chunks(session, doc_id, limit=limit))


@router.post("/{doc_id}/chunk-preview", summary="按新参数预览分块(不落库)")
async def preview_chunks(
    session: SessionDep,
    user_id: CurrentUser,
    doc_id: str,
    payload: Annotated[dict[str, Any] | None, Body()] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> dict[str, Any]:
    """用指定参数重新切分并返回结果, **不写库、不算向量**.

    这是调分块参数的正确姿势: 改参数 → 立刻看到切成什么样 → 满意再应用.
    如果每次都要"改配置 → 重新上传 → 重新向量化"才能看到结果,
    一轮几分钟, 根本没法做参数对比实验.

    解析结果有内存缓存, 所以除了第一次(要解析 PDF), 后续调参都是毫秒级。
    """
    document = await document_service.get_document(session, doc_id, user_id=user_id)

    params = chunk_preview.validate_params_or_raise(payload or {})
    clean = chunk_preview.get_clean_document(doc_id, document.file_path, document.file_md5)

    result = chunk_preview.preview(clean, doc_id, params, limit=limit)
    result["doc_id"] = doc_id
    result["char_count"] = clean.char_count
    result["page_count"] = clean.page_count
    return ok(result)


@router.post("/{doc_id}/chunk-apply", summary="保存分块参数并重新处理")
async def apply_chunk_params(
    session: SessionDep,
    user_id: CurrentUser,
    doc_id: str,
    payload: Annotated[dict[str, Any], Body()],
) -> dict[str, Any]:
    """把预览时满意的参数**保存为全局配置**, 并重新处理这份文档.

    "保存为全局配置"和"立即重跑"是两件事, 一起做是有意的:
    如果只存配置不重跑, 用户会以为已经生效了 —— 但已有文档的分块不会自动变化
    (分块是在入库时确定的), 于是产生"改了参数却没效果"的困惑.

    注意: 只重新处理当前这份文档. 其他已入库的文档仍使用旧参数,
    需要用户在列表里逐个重新处理, 或者后续提供一个"批量重建"入口.
    """
    document = await document_service.get_document(session, doc_id, user_id=user_id)

    params = chunk_preview.validate_params_or_raise(payload)

    from app.services import config_service  # noqa: PLC0415

    config_service.update_runtime_config(
        {
            "chunk_strategy": params.strategy,
            "parent_chunk_size": params.parent_size,
            "child_chunk_size": params.child_size,
            "chunk_overlap": params.overlap,
            "min_chunk_size": params.min_size,
            "chunk_keep_heading": params.keep_heading_in_child,
        }
    )

    from app.models.document import DocumentStatus  # noqa: PLC0415

    document.status = DocumentStatus.PENDING.value
    document.error_msg = None
    await session.commit()

    # 参数变了 → 之前缓存的解析结果虽然还能用(解析不受分块参数影响),
    # 但为了让"重新处理"走一次完整链路, 这里仍然清掉缓存.
    chunk_preview.invalidate(doc_id)

    result = await submit_ingest(doc_id)
    payload_out = IngestStatsOut(**vars(result)).model_dump(mode="json")
    payload_out["applied_params"] = params.to_dict()
    return ok(payload_out)
