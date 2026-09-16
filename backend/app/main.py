"""FastAPI 应用入口.

启动方式::

    cd backend
    python -m uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.router import api_router
from app.core.config import settings
from app.core.exceptions import AppException, ErrorCode
from app.core.logging import get_logger, setup_logging
from app.core.middleware import TraceIdMiddleware
from app.core.response import fail
from app.db.session import dispose_engine, init_db

logger = get_logger("docmind.main")

#: 前端静态资源目录(单页控制台)
STATIC_DIR: Path = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期: 启动时初始化资源, 关闭时优雅释放."""
    setup_logging()
    settings.ensure_dirs()

    logger.info(
        "DocMind 启动中 | env=%s version=%s debug=%s",
        settings.app_env,
        settings.app_version,
        settings.debug,
    )
    logger.info(
        "关键配置 | embedding=%s(%sd) rerank=%s llm=%s@%s task_mode=%s vector_store=chroma/%s",
        settings.embedding_model,
        settings.embedding_dim,
        settings.rerank_model if settings.rerank_enabled else "off",
        settings.llm_model,
        settings.llm_provider,
        settings.task_mode,
        settings.chroma_mode,
    )
    if not settings.llm_configured:
        logger.warning(
            "未检测到 DOCMIND_LLM_API_KEY, 问答接口将返回 503. 请在 %s 中配置后重启.",
            settings.data_dir.parent / ".env",
        )

    # 建表 —— 必须在任何请求到来之前完成, 否则第一个请求会撞上"表不存在"
    await init_db()

    # 应用上次在 Web 界面上保存的配置.
    # 必须在建表/预热**之前**执行: 界面上的配置(如 embedding_device=cuda)
    # 会影响预热时加载模型的设备选择.
    from app.services import config_service  # noqa: PLC0415

    config_service.load_runtime_overrides()

    if settings.warmup_on_startup:
        # 预热本地模型: 把几十秒的加载开销从"第一个用户请求"移到"进程启动".
        # 不预热的话, 第一个提问的用户会看到一次莫名其妙的超时.
        await asyncio.to_thread(warmup_embedding)
    else:
        logger.info("已跳过模型预热(DOCMIND_WARMUP_ON_STARTUP=false)")

    _log_ready_banner()

    yield

    logger.info("DocMind 正在关闭 ...")
    await dispose_engine()
    _shutdown_embedding()
    logger.info("DocMind 已关闭")


def warmup_embedding() -> None:
    """预热 Embedding 模型. 失败只告警不阻塞启动."""
    try:
        from app.services.embedding import warmup  # noqa: PLC0415

        warmup()
    except Exception:  # noqa: BLE001 - 预热失败不该让服务起不来
        logger.exception("Embedding 预热失败, 首次请求会较慢")


def _log_ready_banner() -> None:
    """启动完成后打印入口地址.

    模型预热会占用 20 秒到几分钟, 期间端口还没开始接受连接, 用户只知道"打不开".
    把入口地址在预热**之后**打印出来, 就是给用户一个明确的"现在可以访问了"信号.
    """
    base = f"http://127.0.0.1:{settings.port}"
    logger.info("")
    logger.info("  %s 已就绪, 可以访问以下地址:", settings.app_name)
    logger.info("    Web 控制台  %s/", base)
    logger.info("    接口文档    %s/docs", base)
    logger.info("    就绪探针    %s%s/health/ready", base, settings.api_prefix)
    logger.info("")


def _shutdown_embedding() -> None:
    """释放本地模型占用的显存/内存(P1 阶段实现真正的释放逻辑)."""
    try:
        from app.services.embedding import release_models  # noqa: PLC0415

        release_models()
    except ImportError:
        # P1 之前该模块尚不存在, 属于预期情况
        pass
    except Exception:  # pragma: no cover - 关闭阶段不应因清理失败而中断
        logger.exception("释放本地模型资源时发生异常")


def create_app() -> FastAPI:
    """应用工厂. 便于测试中创建隔离实例."""
    app = FastAPI(
        title=f"{settings.app_name} API",
        description=(
            "面向私有文档的检索增强问答系统: "
            "PDF 解析 -> 父子分块 -> 向量化入库 -> 混合检索 + 重排 -> 引用溯源式生成"
        ),
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(TraceIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Trace-Id"],
    )

    _register_exception_handlers(app)

    # 前端控制台: 单文件 HTML, 不需要 Node、不需要构建步骤.
    # 这样"clone 下来就能用"的门槛最低 —— 直接开浏览器就能操作, 不用先装前端工具链.
    # P6 阶段会被 Vue3 应用替换(或由 Nginx 独立托管).
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    app.include_router(api_router, prefix=settings.api_prefix)

    @app.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        """根路径直接返回 Web 控制台."""
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api", include_in_schema=False)
    async def api_meta() -> dict[str, str]:
        """服务元信息(JSON). 给脚本和监控用, 人类入口在 ``/``."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": f"{settings.api_prefix}/health",
        }

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """全局异常处理: 把各类异常统一翻译成 ApiResponse 结构."""

    @app.exception_handler(AppException)
    async def _handle_app_exception(request: Request, exc: AppException) -> JSONResponse:
        logger.warning(
            "业务异常 | path=%s code=%s message=%s detail=%s",
            request.url.path,
            exc.code.value,
            exc.message,
            exc.detail,
        )
        return JSONResponse(
            status_code=exc.http_status, content=fail(exc.code, exc.message, exc.detail)
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # 把 pydantic 的错误详情压缩成前端友好的结构
        details = [
            {"field": ".".join(str(x) for x in err.get("loc", ())), "reason": err.get("msg", "")}
            for err in exc.errors()
        ]
        logger.warning("参数校验失败 | path=%s details=%s", request.url.path, details)
        return JSONResponse(
            status_code=400,
            content=fail(ErrorCode.PARAM_INVALID, "请求参数不合法", details),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code = {
            401: ErrorCode.UNAUTHORIZED,
            403: ErrorCode.FORBIDDEN,
            404: ErrorCode.NOT_FOUND,
            429: ErrorCode.RATE_LIMITED,
        }.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
        message = exc.detail if isinstance(exc.detail, str) else "请求失败"
        return JSONResponse(status_code=exc.status_code, content=fail(code, message))

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, _exc: Exception) -> JSONResponse:
        # 未预期异常必须打完整堆栈, 但绝不把堆栈返回给前端(信息泄露).
        # 这里用 logger.exception 而非显式打印 _exc: 它在异常处理器中被调用时
        # 能直接取到 sys.exc_info(), 输出完整 traceback.
        logger.exception("未捕获异常 | path=%s", request.url.path)
        trace_id = getattr(request.state, "trace_id", "-")
        return JSONResponse(
            status_code=500,
            content=fail(
                ErrorCode.INTERNAL_ERROR,
                "服务内部错误, 请稍后重试",
                {"trace_id": trace_id} if settings.debug else None,
            ),
        )


app = create_app()
