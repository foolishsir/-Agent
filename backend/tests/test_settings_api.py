"""运行时配置接口与配置服务的测试.

重点覆盖三件事:
1. **密钥永不回显** —— 脱敏是安全底线
2. **跨字段一致性校验** —— 单字段合法但组合非法的配置必须被拦住
3. **改完立即生效** —— 配置改了但缓存没失效, 是最典型的"改了没用"故障
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.exceptions import ParamInvalidError
from app.services import config_service


@pytest.fixture(autouse=True)
def _isolate_settings() -> Iterator[None]:
    """每个用例前后快照 / 还原可编辑配置, 并清掉落盘文件.

    为什么必须做: ``settings`` 是**进程级单例**, 直接改它会让用例互相污染 ——
    前一个用例把 ``vector_top_k`` 改成 5, 后一个用例的跨字段一致性校验
    (final_top_k 不能超过召回条数) 就会意外失败.

    这类"顺序相关的测试失败"极其难排查, 因为它单独跑是过的、一起跑才挂.
    正确做法是让每个用例从一个干净的配置开始, 而不是去调整用例顺序.
    """
    snapshot = {f.key: getattr(settings, f.key) for f in config_service.CONFIG_FIELDS}
    path = config_service.runtime_config_path()
    if path.exists():
        path.unlink()

    yield

    for key, value in snapshot.items():
        setattr(settings, key, value)
    if path.exists():
        path.unlink()


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
def test_get_config_returns_grouped_fields(client: TestClient) -> None:
    data = client.get("/api/v1/settings").json()["data"]

    assert data["groups"]
    names = {g["name"] for g in data["groups"]}
    assert {"大模型", "检索", "分块"} <= names

    keys = {f["key"] for g in data["groups"] for f in g["fields"]}
    assert {"llm_api_key", "llm_model", "vector_top_k", "final_top_k"} <= keys


def test_field_metadata_is_sent_to_frontend(client: TestClient) -> None:
    """前端靠后端下发的元数据动态渲染表单, 新增配置项不用改前端."""
    data = client.get("/api/v1/settings").json()["data"]
    fields = {f["key"]: f for g in data["groups"] for f in g["fields"]}

    temperature = fields["llm_temperature"]
    assert temperature["type"] == "float"
    assert temperature["min"] == 0.0
    assert temperature["max"] == 2.0
    assert temperature["description"]

    assert fields["llm_provider"]["options"]
    assert fields["child_chunk_size"]["requires_reindex"] is True


def test_api_key_is_masked_in_response(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """密钥**永远**不回显完整值 —— 即使接口没有鉴权, 也不能泄露出去."""
    monkeypatch.setattr(settings, "llm_api_key", "sk-super-secret-key-1234")

    data = client.get("/api/v1/settings").json()["data"]
    field = next(f for g in data["groups"] for f in g["fields"] if f["key"] == "llm_api_key")

    assert field["value"] == "********"
    assert field["configured"] is True
    assert "1234" in field["hint"]
    # 完整密钥绝不能出现在响应体里
    assert "sk-super-secret-key-1234" not in client.get("/api/v1/settings").text


# --------------------------------------------------------------------------- #
# 更新
# --------------------------------------------------------------------------- #
def test_update_applies_immediately(client: TestClient) -> None:
    resp = client.put("/api/v1/settings", json={"vector_top_k": 33})
    assert resp.status_code == 200
    assert settings.vector_top_k == 33

    # 再次读取应该看到新值
    data = client.get("/api/v1/settings").json()["data"]
    field = next(f for g in data["groups"] for f in g["fields"] if f["key"] == "vector_top_k")
    assert field["value"] == 33


def test_empty_secret_means_keep_existing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """密钥字段留空表示"不修改", 而不是"清空".

    否则用户只想调个温度, 却会把已保存的 Key 一起抹掉 —— 这是很反直觉的行为.
    """
    monkeypatch.setattr(settings, "llm_api_key", "sk-existing-key")

    client.put("/api/v1/settings", json={"llm_api_key": "", "llm_temperature": 0.3})

    assert settings.llm_api_key == "sk-existing-key"
    assert settings.llm_temperature == 0.3


def test_masked_secret_roundtrip_does_not_corrupt_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """前端把脱敏值(********)原样提交回来时, 不能把它当成新密钥写进去."""
    monkeypatch.setattr(settings, "llm_api_key", "sk-real-key")

    client.put("/api/v1/settings", json={"llm_api_key": "********"})

    assert settings.llm_api_key == "sk-real-key"


def test_unknown_field_is_rejected(client: TestClient) -> None:
    resp = client.put("/api/v1/settings", json={"not_a_real_field": 1})
    assert resp.status_code == 400
    assert resp.json()["code"] == "PARAM_INVALID"


def test_out_of_range_value_is_rejected(client: TestClient) -> None:
    before = settings.llm_temperature
    resp = client.put("/api/v1/settings", json={"llm_temperature": 99})
    assert resp.status_code == 400
    assert settings.llm_temperature == before, "校验失败时不能留下半改状态"


def test_non_numeric_value_is_rejected(client: TestClient) -> None:
    resp = client.put("/api/v1/settings", json={"vector_top_k": "abc"})
    assert resp.status_code == 400


def test_boolean_is_coerced(client: TestClient) -> None:
    client.put("/api/v1/settings", json={"rerank_enabled": "false"})
    assert settings.rerank_enabled is False
    client.put("/api/v1/settings", json={"rerank_enabled": True})
    assert settings.rerank_enabled is True


# --------------------------------------------------------------------------- #
# 跨字段一致性
# --------------------------------------------------------------------------- #
def test_child_chunk_must_be_smaller_than_parent(client: TestClient) -> None:
    """子块比父块还大时父子结构失去意义.

    这类错误如果不拦, **不会报错**, 只会静默产出垃圾结果 —— 比直接报错更难排查.
    """
    with pytest.raises(ParamInvalidError, match="子块大小必须小于父块大小"):
        config_service.update_runtime_config({"parent_chunk_size": 300, "child_chunk_size": 500})


def test_rejected_update_leaves_no_partial_state(client: TestClient) -> None:
    """被拒绝的更新必须**完整回滚**, 不能留下半改状态.

    这是一个真实踩到的 bug: 最初的实现是"逐字段 setattr → 最后统一做跨字段校验",
    于是校验失败时非法值**已经写进 settings 单例了**.
    接口返回 400 看起来一切正常, 但残留的 parent=300/child=500
    让**之后每一次文档上传都失败**, 而错误信息完全指向不到那次被拒绝的请求.

    这类"校验通过但状态已脏"的 bug 极难排查, 因为故障发生在另一个请求上,
    现场与根因隔得很远. 正确做法就是"先算全部目标值 → 整体应用 → 失败则回滚".
    """
    before_parent = settings.parent_chunk_size
    before_child = settings.child_chunk_size

    with pytest.raises(ParamInvalidError):
        config_service.update_runtime_config({"parent_chunk_size": 300, "child_chunk_size": 500})

    # 两个字段都必须回到原值 —— 一个都不能残留
    assert settings.parent_chunk_size == before_parent
    assert settings.child_chunk_size == before_child


def test_rejected_update_does_not_persist(client: TestClient) -> None:
    """被拒绝的更新不能落盘, 否则重启后非法配置会"复活"."""
    path = config_service.runtime_config_path()

    with pytest.raises(ParamInvalidError):
        config_service.update_runtime_config({"parent_chunk_size": 300, "child_chunk_size": 500})

    assert not path.exists() or '"parent_chunk_size": 300' not in path.read_text(encoding="utf-8")


def test_valid_update_still_works_after_rejection(client: TestClient) -> None:
    """被拒绝之后合法更新依然要能工作 —— 回滚逻辑不能把服务弄坏."""
    with pytest.raises(ParamInvalidError):
        config_service.update_runtime_config({"parent_chunk_size": 300, "child_chunk_size": 500})

    changed = config_service.update_runtime_config({"parent_chunk_size": 1600})
    assert changed.get("parent_chunk_size") == 1600
    assert settings.parent_chunk_size == 1600


def test_final_top_k_cannot_exceed_recall(client: TestClient) -> None:
    with pytest.raises(ParamInvalidError, match="进入 Prompt 的条数"):
        config_service.update_runtime_config(
            {"vector_top_k": 5, "bm25_top_k": 0, "final_top_k": 20}
        )


def test_overlap_cannot_exceed_child_size(client: TestClient) -> None:
    with pytest.raises(ParamInvalidError, match="子块重叠"):
        config_service.update_runtime_config({"child_chunk_size": 100, "chunk_overlap": 150})


# --------------------------------------------------------------------------- #
# 持久化
# --------------------------------------------------------------------------- #
def test_config_is_persisted_to_disk(client: TestClient) -> None:
    client.put("/api/v1/settings", json={"rrf_k": 42})

    path = config_service.runtime_config_path()
    assert path.exists()
    assert '"rrf_k": 42' in path.read_text(encoding="utf-8")


def test_reload_restores_saved_values(client: TestClient) -> None:
    """重启后能读回上次保存的配置 —— 否则"在界面上配好"这件事就没有意义."""
    config_service.update_runtime_config({"rrf_k": 55})
    settings.rrf_k = 1  # 模拟重启后回到默认值

    applied = config_service.load_runtime_overrides()

    assert applied > 0
    assert settings.rrf_k == 55


def test_corrupt_config_file_does_not_break_startup(client: TestClient) -> None:
    """配置文件损坏时**不能**让服务起不来 —— 用默认值继续跑并告警即可."""
    config_service.runtime_config_path().write_text("{ this is not json", encoding="utf-8")

    assert config_service.load_runtime_overrides() == 0


def test_reset_removes_override_file(client: TestClient) -> None:
    config_service.update_runtime_config({"rrf_k": 44})
    assert config_service.runtime_config_path().exists()

    client.delete("/api/v1/settings")

    assert not config_service.runtime_config_path().exists()


# --------------------------------------------------------------------------- #
# 缓存失效
# --------------------------------------------------------------------------- #
def test_llm_client_cache_is_invalidated_on_key_change(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """改了 Key 但客户端还是旧的 → 用户会以为"界面上改了却没生效"."""
    from app.services import llm

    calls: list[int] = []
    monkeypatch.setattr(llm, "reset_llm_client", lambda: calls.append(1))

    config_service._invalidate_caches({"llm_api_key"})

    assert calls, "修改 LLM 相关配置后必须重置客户端缓存"


def test_runtime_config_disabled_blocks_writes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """线上部署可以整体关掉该功能, 改用环境变量配置."""
    monkeypatch.setattr(settings, "allow_runtime_config", False)

    resp = client.put("/api/v1/settings", json={"vector_top_k": 10})
    assert resp.status_code == 400
    assert "禁用" in resp.json()["message"]
