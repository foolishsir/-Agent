"""HTTP 中间件.

包含两个能力, 合并为一个中间件以降低调用链开销:

1. **TraceId 注入** —— 优先复用上游传入的 ``X-Trace-Id``(便于网关/前端串联),
   否则新生成一个; 同时写回响应头, 前端排查问题时可以直接把 id 发给后端.
2. **访问日志** —— 记录 method / path / status / 耗时, 慢请求单独告警.

注意: SSE 流式接口的耗时会被完整记录, 这是有意的 —— 首 token 延迟(P95)
是 RAG 系统最重要的性能指标之一, 后续在评测阶段会从这里采集.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger, set_trace_id

logger = get_logger("docmind.access")

TRACE_HEADER = "X-Trace-Id"

# 超过该耗时的请求打 WARNING, 便于快速定位性能劣化
SLOW_REQUEST_MS = 3000.0

# 健康检查/文档等高频噪音路径不记录访问日志
_SILENT_PATHS = ("/docs", "/redoc", "/openapi.json", "/favicon.ico")


class TraceIdMiddleware(BaseHTTPMiddleware):
    """链路 id 与访问日志中间件."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        trace_id = set_trace_id(request.headers.get(TRACE_HEADER))
        request.state.trace_id = trace_id

        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[TRACE_HEADER] = trace_id
            return response
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            path = request.url.path
            if path not in _SILENT_PATHS:
                record = logger.warning if elapsed_ms >= SLOW_REQUEST_MS else logger.info
                record(
                    "%s %s -> %s | cost_ms=%.1f trace_id=%s client=%s",
                    request.method,
                    path,
                    status_code,
                    elapsed_ms,
                    trace_id,
                    request.client.host if request.client else "-",
                )
