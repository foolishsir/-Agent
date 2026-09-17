"""健康检查与就绪探针.

区分两个语义(这是 K8s 部署的标准做法, 面试常问):

- ``/health``      **存活探针**: 只证明进程没死, 不检查依赖. 失败 => 重启容器.
- ``/health/ready`` **就绪探针**: 检查关键依赖是否可用. 失败 => 摘除流量, 不重启.

就绪检查会真实探测依赖(而不是返回硬编码的 ok), 这在排查
「服务起来了但一问答就 500」这类问题时非常有用.
"""

from __future__ import annotations

import importlib.util
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter

from app.core.config import settings
from app.core.logging import get_logger
from app.core.response import ok

logger = get_logger("docmind.health")

router = APIRouter()


@router.get("", summary="存活探针")
async def liveness() -> dict[str, Any]:
    """进程存活检查. 不触碰任何外部依赖, 必须足够快."""
    return ok(
        {
            "status": "up",
            "app": settings.app_name,
            "version": settings.app_version,
            "env": settings.app_env,
            "time": datetime.now(UTC).isoformat(),
        }
    )


def _check_package(module_name: str) -> tuple[bool, str]:
    """检查 Python 包是否可导入(不实际加载, 避免拖慢探针)."""
    found = importlib.util.find_spec(module_name) is not None
    return found, "已安装" if found else f"未安装 ({module_name})"


def _check_writable_dirs() -> tuple[bool, str]:
    try:
        settings.ensure_dirs()
        probe = settings.data_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, str(settings.data_dir)
    except Exception as exc:  # pragma: no cover - 依赖文件系统状态
        return False, f"目录不可写: {exc}"


def _check_llm() -> tuple[bool, str]:
    """只检查"配没配", **不检查"能不能用"**.

    探针刻意不发真实请求: `/health/ready` 会被前端每 30 秒轮询一次,
    每次都调一次模型既慢又费钱.

    代价是存在盲区 —— **Key 过期/被吊销时这里仍然显示就绪**.
    这个盲区是真实踩到的: Key 401 之后探针一路绿灯, 直到提问才报错.
    所以 detail 里写清 "仅检查是否配置", 避免它给出虚假的安全感.
    要验证 Key 是否真能用, 走「设置」页的**测试连接**(那里是真的调一次模型).
    """
    if not settings.llm_configured:
        return False, "未配置 DOCMIND_LLM_API_KEY"
    key = settings.llm_api_key
    masked = f"{key[:4]}***{key[-4:]}" if len(key) > 8 else "***"
    return True, f"{settings.llm_provider}/{settings.llm_model} key={masked} (仅检查是否配置)"


def _check_embedding() -> tuple[bool, str]:
    pkg = "sentence_transformers" if settings.embedding_provider == "local" else "openai"
    installed, detail = _check_package(pkg)
    if not installed:
        return False, detail
    return (
        True,
        f"{settings.embedding_provider}/{settings.embedding_model} dim={settings.embedding_dim}",
    )


def _check_vector_store() -> tuple[bool, str]:
    installed, detail = _check_package("chromadb")
    if not installed:
        return False, detail
    if settings.chroma_mode == "http":
        return True, f"http://{settings.chroma_host}:{settings.chroma_port}"
    return True, f"embedded @ {settings.chroma_persist_dir}"


def _check_rerank() -> tuple[bool, str]:
    if not settings.rerank_enabled or settings.rerank_provider == "none":
        return True, "已关闭"
    return _check_package("sentence_transformers")[0], settings.rerank_model


def _check_redis() -> tuple[bool, str]:
    if settings.task_mode != "queue":
        return True, "inline 模式, 未启用"
    return _check_package("redis"), settings.redis_url


async def _check_database() -> tuple[bool, str]:
    """真实连一次数据库, 而不是只看配置.

    价值: 服务能起来但数据库文件被锁、路径不可写时, 就绪探针能第一时间暴露,
    而不是等用户上传文档才 500.
    """
    try:
        from app.db.session import check_connection  # noqa: PLC0415

        return await check_connection()
    except Exception as exc:  # noqa: BLE001 - 探针需要兜住一切
        return False, f"数据库不可用: {exc}"


@router.get("/ready", summary="就绪探针")
async def readiness() -> dict[str, Any]:
    """检查关键依赖, 返回逐项明细, 便于定位是哪个环节没准备好."""
    started = time.perf_counter()

    checks: dict[str, tuple[bool, str]] = {
        "workspace": _check_writable_dirs(),
        "database": await _check_database(),
        "llm": _check_llm(),
        "embedding": _check_embedding(),
        "vector_store": _check_vector_store(),
        "rerank": _check_rerank(),
        "redis": _check_redis(),
    }

    details = {
        name: {"ready": passed, "detail": detail} for name, (passed, detail) in checks.items()
    }
    # llm 未配置不算「未就绪」: 文档上传/入库链路依然可用, 只有问答不可用.
    # 这样避免探针把整个服务摘掉, 但问题依然能在明细里被看见.
    blocking = ("workspace", "database", "embedding", "vector_store")
    ready = all(details[name]["ready"] for name in blocking)

    return ok(
        {
            "status": "ready" if ready else "not_ready",
            "ready": ready,
            "blocking": list(blocking),
            "checks": details,
            "cost_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    )
