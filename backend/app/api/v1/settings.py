"""运行时配置接口.

让使用者可以在 **Web 界面上直接配置** API Key、模型名、检索参数,
改完立即生效, 不需要编辑 .env 文件, 也不需要重启服务.

安全提示
--------
这些接口**没有身份校验**(项目本身也没有登录体系), 所以:

- 密钥类字段在**读取时永远脱敏**(只回显末四位), 不会泄露出去
- 但**写入**是开放的 —— 任何能访问该服务的人都能改配置

因此线上部署必须:
1. 设置 ``DOCMIND_ALLOW_RUNTIME_CONFIG=false`` 关掉该功能, 改用环境变量; 或
2. 在 Nginx / 网关层加 Basic Auth 或 IP 白名单

这是"开箱易用"与"安全默认"之间的显式取舍. 选择默认开启,
是因为本地开发场景下让使用者去翻 .env 文件的体验太差;
而线上场景一定会有网关, 由网关兜住鉴权是更合理的分层.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core.exceptions import ParamInvalidError
from app.core.response import ok
from app.services import config_service

router = APIRouter()


@router.get("", summary="读取当前配置")
async def get_config() -> dict[str, Any]:
    """返回可编辑的配置项(密钥脱敏)及字段元数据.

    元数据(type/options/min/max/description)由后端下发, 前端据此动态渲染表单.
    这样做的好处是: 新增一个配置项只需要改后端, 前端不用动一行代码.
    """
    return ok(config_service.build_config_view())


@router.put("", summary="更新配置")
async def update_config(payload: dict[str, Any]) -> dict[str, Any]:
    """批量更新配置项, 改完**立即生效**.

    只提交需要改的字段即可. 密钥字段留空表示"不修改",
    避免用户只想调温度却把已保存的 Key 一起清掉.
    """
    if not isinstance(payload, dict):
        raise ParamInvalidError("请求体必须是 JSON 对象")

    changed = config_service.update_runtime_config(payload)
    return ok(
        {
            "changed": changed,
            "count": len(changed),
            "message": "配置已更新并生效" if changed else "没有需要更新的字段",
            "config": config_service.build_config_view(),
        }
    )


@router.delete("", summary="重置配置")
async def reset_config() -> dict[str, Any]:
    """清空界面上的配置, 回退到 .env / 代码默认值(需重启生效)."""
    return ok(config_service.reset_runtime_config())


@router.post("/test-llm", summary="测试大模型连接")
async def test_llm(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """用当前配置真实调用一次大模型, 验证 Key / 地址 / 模型名是否可用.

    为什么不只做格式校验: "Key 格式看起来对"和"Key 真的能用"是两回事 ——
    余额不足、模型名写错、地址不通都会在真实调用时才暴露.
    与其等用户去提问才报错, 不如提供一键自检.
    """
    overrides = payload or {}
    if overrides:
        config_service.update_runtime_config(overrides)

    from app.services.llm import test_connection  # noqa: PLC0415 - P2 之后才有

    return ok(await test_connection())
