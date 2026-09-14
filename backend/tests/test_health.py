"""健康检查与全局异常处理的行为约定测试."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_returns_service_meta(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "DocMind"
    assert "version" in body


def test_liveness_ok(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == "OK"
    assert body["data"]["status"] == "up"


def test_readiness_reports_each_dependency(client: TestClient) -> None:
    """就绪探针必须逐项返回依赖状态, 而不是笼统一个 ok."""
    resp = client.get("/api/v1/health/ready")
    assert resp.status_code == 200
    data = resp.json()["data"]

    assert set(data["checks"]) >= {
        "workspace",
        "llm",
        "embedding",
        "vector_store",
        "rerank",
        "redis",
    }
    # 目录可写属于阻塞项, 测试环境必须通过
    assert data["checks"]["workspace"]["ready"] is True
    assert isinstance(data["ready"], bool)


def test_trace_id_header_is_echoed(client: TestClient) -> None:
    """回传 trace_id 是排障的基础能力, 必须有."""
    resp = client.get("/api/v1/health", headers={"X-Trace-Id": "test-trace-001"})
    assert resp.headers.get("X-Trace-Id") == "test-trace-001"


def test_trace_id_generated_when_absent(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    trace_id = resp.headers.get("X-Trace-Id")
    assert trace_id and len(trace_id) == 12


def test_response_envelope_is_uniform(client: TestClient) -> None:
    """所有接口共用同一套响应外壳, 前端才能写统一拦截器."""
    body = client.get("/api/v1/health").json()
    assert set(body) >= {"code", "message", "data", "trace_id"}


def test_404_uses_unified_error_body(client: TestClient) -> None:
    resp = client.get("/api/v1/not-exist-endpoint")
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "NOT_FOUND"
    assert body["data"] is None
    assert "trace_id" in body
