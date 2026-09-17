"""运行时配置服务 —— 支持在 Web 界面里直接修改配置, 而不是去编辑 .env.

设计取舍
--------
**为什么需要它**: `.env` 适合"部署时确定、之后不变"的配置(端口、路径、数据库地址).
但 LLM API Key、模型名、检索参数这类东西, 使用者希望在界面上点几下就能改,
改完立即生效, 不用重启服务、也不用去翻文件.

**配置优先级**: 运行时覆盖 > 环境变量 / .env > 代码默认值
这与 12-Factor App 的原则一致 —— 环境变量永远高于代码默认值,
而用户在界面上的显式修改高于环境变量(否则界面上改了却不生效, 是更糟的体验).

**持久化**: 存成 ``data/runtime_settings.json``, 而不是数据库表.
理由: 配置读取非常频繁(几乎每次 LLM 调用都要读), 走文件缓存到内存最直接;
而配置项数量固定、结构简单, 用不上 SQL 的查询能力. 换成 DB 只会增加一次 I/O.

**安全**: 密钥类字段在读取时**永远脱敏**, 只返回是否已配置和末四位.
即使这个接口没有鉴权, 也不会把密钥回显出去. 但要注意:
这仍然意味着**任何能访问该接口的人都能改写配置**, 所以线上部署必须
在网关层加鉴权, 或者关掉这个功能(见 ``allow_runtime_config`` 开关).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.exceptions import ParamInvalidError
from app.core.logging import get_logger

logger = get_logger("docmind.config")


# --------------------------------------------------------------------------- #
# 字段定义
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConfigField:
    """一个可在界面上编辑的配置项."""

    key: str
    label: str
    group: str
    type: str  # str | int | float | bool | secret | select
    description: str = ""
    options: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    #: 改动后需要重建索引才生效(如分块参数、Embedding 模型)
    requires_reindex: bool = False
    #: 改动后需要重启服务才生效
    requires_restart: bool = False


CONFIG_FIELDS: tuple[ConfigField, ...] = (
    # ---------------------------- 大模型 ----------------------------
    ConfigField(
        key="llm_provider",
        label="服务商",
        group="大模型",
        type="select",
        options=("deepseek", "openai", "siliconflow", "dashscope", "custom"),
        description="仅作标识用, 实际请求走下面的 base_url",
    ),
    ConfigField(
        key="llm_base_url",
        label="API 地址",
        group="大模型",
        type="str",
        description="OpenAI 兼容协议地址, 需带 /v1",
    ),
    ConfigField(
        key="llm_api_key",
        label="API Key",
        group="大模型",
        type="secret",
        description="必填。未配置时问答接口会返回 503",
    ),
    ConfigField(
        key="llm_model",
        label="模型名称",
        group="大模型",
        type="str",
        description="如 deepseek-chat / qwen-plus / gpt-4o-mini",
    ),
    ConfigField(
        key="llm_temperature",
        label="温度",
        group="大模型",
        type="float",
        minimum=0.0,
        maximum=2.0,
        description="问答场景建议 0~0.3, 越低越稳定、越少幻觉",
    ),
    ConfigField(
        key="llm_max_tokens",
        label="最大输出长度",
        group="大模型",
        type="int",
        minimum=128,
        maximum=8192,
    ),
    # ---------------------------- 检索 ----------------------------
    ConfigField(
        key="vector_top_k",
        label="向量召回条数",
        group="检索",
        type="int",
        minimum=1,
        maximum=100,
    ),
    ConfigField(
        key="bm25_top_k",
        label="关键词召回条数",
        group="检索",
        type="int",
        minimum=0,
        maximum=100,
        description="设为 0 表示关闭 BM25, 退化成纯向量检索",
    ),
    ConfigField(
        key="rrf_k", label="RRF 融合常数", group="检索", type="int", minimum=1, maximum=200
    ),
    ConfigField(
        key="final_top_k",
        label="进入 Prompt 的条数",
        group="检索",
        type="int",
        minimum=1,
        maximum=20,
        description="精排后最终送给大模型的文档片段数",
    ),
    ConfigField(
        key="rerank_enabled",
        label="启用重排",
        group="检索",
        type="bool",
        description="Cross-Encoder 精排, 精度提升最大的一环",
    ),
    ConfigField(
        key="rerank_min_score",
        label="拒答阈值",
        group="检索",
        type="float",
        minimum=0.0,
        maximum=1.0,
        description="精排分数低于此值时判定文档中无相关内容, 直接拒答。0 表示不拒答",
    ),
    # ---------------------------- 分块 ----------------------------
    ConfigField(
        key="chunk_strategy",
        label="分块策略",
        group="分块",
        type="select",
        options=("parent_child", "recursive", "fixed"),
        requires_reindex=True,
        description="parent_child 父子块(推荐) / recursive 递归字符切分 / fixed 固定长度(基线)",
    ),
    ConfigField(
        key="parent_chunk_size",
        label="父块大小",
        group="分块",
        type="int",
        minimum=100,
        maximum=4000,
        requires_reindex=True,
        description="喂给大模型的上下文单位",
    ),
    ConfigField(
        key="child_chunk_size",
        label="子块大小",
        group="分块",
        type="int",
        minimum=20,
        maximum=1200,
        requires_reindex=True,
        description="用于向量检索的单位, 必须小于父块",
    ),
    ConfigField(
        key="chunk_overlap",
        label="子块重叠",
        group="分块",
        type="int",
        minimum=0,
        maximum=300,
        requires_reindex=True,
    ),
    ConfigField(
        key="min_chunk_size",
        label="最小块长度",
        group="分块",
        type="int",
        minimum=0,
        maximum=200,
        requires_reindex=True,
        description="短于此值的块会被并入相邻块, 避免产生无信息量的向量",
    ),
    ConfigField(
        key="chunk_separators",
        label="递归分隔符",
        group="分块",
        type="str",
        requires_reindex=True,
        description="仅 recursive 策略生效。按优先级排列, 用 | 分隔, 换行写 \\n",
    ),
    ConfigField(
        key="chunk_keep_heading",
        label="标题保留在子块正文",
        group="分块",
        type="bool",
        requires_reindex=True,
        description="开启后子块正文自带章节标题; 关闭则靠 embedding_text 补语境(推荐关闭)",
    ),
    # ---------------------------- 向量模型 ----------------------------
    ConfigField(
        key="embedding_device",
        label="向量模型设备",
        group="向量模型",
        type="select",
        options=("cpu", "cuda"),
        requires_restart=True,
        description="有 NVIDIA 显卡可选 cuda",
    ),
    ConfigField(
        key="rerank_device",
        label="重排模型设备",
        group="向量模型",
        type="select",
        options=("cpu", "cuda"),
        requires_restart=True,
    ),
    ConfigField(
        key="embedding_batch_size",
        label="向量化批大小",
        group="向量模型",
        type="int",
        minimum=1,
        maximum=256,
    ),
    # ---------------------------- 语音 ----------------------------
    ConfigField(
        key="speech_asr_provider",
        label="语音输入 (ASR)",
        group="语音",
        type="select",
        options=("dashscope", "none"),
        description="dashscope = 阿里云 Paraformer, 中文效果最好; 需要下面的百炼 Key",
    ),
    ConfigField(
        key="dashscope_api_key",
        label="阿里云百炼 Key",
        group="语音",
        type="secret",
        description="仅语音识别用。和上面「大模型 API Key」不是一个东西",
    ),
    ConfigField(
        key="speech_asr_model",
        label="识别模型",
        group="语音",
        type="str",
        description="paraformer-realtime-v2 / paraformer-realtime-8k-v2(电话音质)",
    ),
    ConfigField(
        key="speech_tts_provider",
        label="语音输出 (TTS)",
        group="语音",
        type="select",
        options=("edge", "none"),
        description="edge = edge-tts, 免费且无需 Key, 需要联网",
    ),
    ConfigField(
        key="speech_tts_voice",
        label="发音人",
        group="语音",
        type="select",
        options=(
            "zh-CN-XiaoxiaoNeural",
            "zh-CN-XiaoyiNeural",
            "zh-CN-YunxiNeural",
            "zh-CN-YunjianNeural",
            "zh-CN-YunyangNeural",
            "zh-CN-liaoning-XiaobeiNeural",
            "zh-CN-shaanxi-XiaoniNeural",
        ),
        description="Xiaoxiao 女声自然 / Yunxi 男声沉稳 / Yunyang 播报腔",
    ),
    ConfigField(
        key="speech_tts_rate",
        label="语速",
        group="语音",
        type="str",
        description="形如 -5% / +10%。面试提问稍慢一点更好听清",
    ),
    ConfigField(
        key="speech_max_audio_seconds",
        label="单段录音上限(秒)",
        group="语音",
        type="int",
        minimum=10,
        maximum=600,
        description="超过会被前端自动截断",
    ),
)

_FIELD_MAP: dict[str, ConfigField] = {f.key: f for f in CONFIG_FIELDS}

_SECRET_MASK = "********"


# --------------------------------------------------------------------------- #
# 读写
# --------------------------------------------------------------------------- #
def runtime_config_path() -> Path:
    return settings.data_dir / "runtime_settings.json"


def load_runtime_overrides() -> int:
    """启动时把上次保存的界面配置应用回 settings 单例.

    返回成功应用的项数. 文件不存在或损坏时**不抛异常** ——
    配置读不出来不应该让服务起不来, 用默认值继续跑并告警即可.
    """
    path = runtime_config_path()
    if not path.exists():
        return 0

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("运行时配置文件损坏, 已忽略 | path=%s error=%s", path, exc)
        return 0

    applied = 0
    for key, value in raw.items():
        field_def = _FIELD_MAP.get(key)
        if field_def is None:
            continue
        try:
            setattr(settings, key, _coerce(field_def, value))
            applied += 1
        except (ParamInvalidError, ValueError, TypeError) as exc:
            logger.warning("跳过无效的运行时配置 | %s=%r error=%s", key, value, exc)

    if applied:
        logger.info("已应用 %s 项界面配置 | path=%s", applied, path)
    return applied


def update_runtime_config(updates: dict[str, Any]) -> dict[str, Any]:
    """更新配置: 校验 → 原子应用 → 落盘 → 失效相关缓存.

    **为什么必须"先全部校验, 再统一应用, 失败则回滚"**

    最初的实现是「逐字段 setattr → 最后统一做跨字段一致性校验」, 结果踩了坑:
    当跨字段校验失败时, 非法值**已经写进 settings 单例了** ——
    接口返回了 400, 但内存里的配置已经被污染.

    真实后果: 测试时提交了 ``parent_chunk_size=300, child_chunk_size=500``,
    接口正确返回 400, 但残留的非法值让**之后每一次文档上传都失败**,
    而且错误信息指向的是"分块参数不合法", 完全联想不到是几天前那次被拒绝的请求留下的.

    这类"校验通过但状态已脏"的 bug 极难排查, 因为:
    - 出错的请求返回了正确的结果, 看起来一切正常
    - 故障在**之后的另一个请求**才爆发, 且现场与根因隔了很远

    正确做法就是这里的模式: 先算出全部目标值, 再整体应用,
    校验不通过就完整回滚. 要么全成功, 要么状态不变.
    """
    if not settings.allow_runtime_config:
        raise ParamInvalidError(
            "运行时配置已被禁用(DOOMIND_ALLOW_RUNTIME_CONFIG=false), 请改用环境变量配置"
        )

    unknown = set(updates) - set(_FIELD_MAP)
    if unknown:
        raise ParamInvalidError(f"不支持修改以下配置项: {', '.join(sorted(unknown))}")

    # ---------------- ① 先算出全部目标值(不接触 settings) ----------------
    pending: dict[str, Any] = {}
    for key, raw_value in updates.items():
        field_def = _FIELD_MAP[key]

        # 前端把密钥留空表示"不修改", 而不是"清空".
        # 否则用户只想改温度, 却会把已保存的 Key 一起抹掉.
        if field_def.type == "secret" and raw_value in ("", _SECRET_MASK, None):
            continue

        value = _coerce(field_def, raw_value)
        if getattr(settings, key, None) != value:
            pending[key] = value

    if not pending:
        return {}

    # ---------------- ② 应用 + 校验, 失败时回滚 ----------------
    # 快照只覆盖受影响的字段: 回滚范围最小化, 也避免误改无关配置
    snapshot = {key: getattr(settings, key, None) for key in pending}

    for key, value in pending.items():
        setattr(settings, key, value)

    try:
        _validate_consistency()
    except ParamInvalidError:
        for key, original in snapshot.items():
            setattr(settings, key, original)
        logger.warning("配置更新被拒绝, 已回滚 | fields=%s", ", ".join(sorted(pending)))
        raise

    # ---------------- ③ 落盘 + 失效缓存 ----------------
    changed = {
        key: _SECRET_MASK if _FIELD_MAP[key].type == "secret" else value
        for key, value in pending.items()
    }
    _persist()
    _invalidate_caches(set(changed))
    logger.info("配置已更新 | fields=%s", ", ".join(sorted(changed)))

    return changed


def reset_runtime_config() -> dict[str, Any]:
    """清空界面配置, 回退到 .env / 代码默认值.

    只删除覆盖文件并重启生效 —— 已 ``setattr`` 到单例上的值无法可靠还原
    (因为单例不记录"原始值"). 这是一个有意的简化: 与其实现一套
    "记住原始值再还原"的机制, 不如让用户重启一次, 语义更清晰.
    """
    path = runtime_config_path()
    if path.exists():
        path.unlink()
    return {"removed": str(path), "note": "需要重启服务后生效"}


# --------------------------------------------------------------------------- #
# 视图
# --------------------------------------------------------------------------- #
def build_config_view() -> dict[str, Any]:
    """构造给前端的配置视图(密钥脱敏 + 附带字段元数据)."""
    groups: dict[str, list[dict[str, Any]]] = {}

    for field_def in CONFIG_FIELDS:
        value = getattr(settings, field_def.key, None)
        item: dict[str, Any] = {
            "key": field_def.key,
            "label": field_def.label,
            "type": field_def.type,
            "description": field_def.description,
            "requires_reindex": field_def.requires_reindex,
            "requires_restart": field_def.requires_restart,
        }

        if field_def.type == "secret":
            configured = bool(str(value or "").strip())
            item["value"] = _SECRET_MASK if configured else ""
            item["configured"] = configured
            # 只回显末四位, 让用户确认"填的是哪把钥匙", 又不泄露完整密钥
            if configured and len(str(value)) >= 8:
                item["hint"] = f"已配置, 末四位 {str(value)[-4:]}"
            else:
                item["hint"] = "未配置" if not configured else "已配置"
        else:
            item["value"] = value
            item["configured"] = True

        if field_def.options:
            item["options"] = list(field_def.options)
        if field_def.minimum is not None:
            item["min"] = field_def.minimum
        if field_def.maximum is not None:
            item["max"] = field_def.maximum

        groups.setdefault(field_def.group, []).append(item)

    return {
        "groups": [{"name": name, "fields": fields} for name, fields in groups.items()],
        "runtime_config_path": str(runtime_config_path()),
        "runtime_config_enabled": settings.allow_runtime_config,
    }


# --------------------------------------------------------------------------- #
# 内部
# --------------------------------------------------------------------------- #
def _coerce(field_def: ConfigField, value: Any) -> Any:
    """按字段类型转换并做范围校验.

    必须显式校验: pydantic 的 ``BaseSettings`` 默认不开 ``validate_assignment``,
    也就是说 ``setattr`` 走的是"直接赋值"路径, 不会触发类型校验.
    不在这里兜住, 一个字符串 "abc" 就能被塞进 int 字段, 直到用的时候才炸.
    """
    if field_def.type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    if field_def.type in ("int", "float"):
        try:
            number = int(value) if field_def.type == "int" else float(value)
        except (TypeError, ValueError) as exc:
            raise ParamInvalidError(f"{field_def.label} 必须是数字") from exc

        if field_def.minimum is not None and number < field_def.minimum:
            raise ParamInvalidError(f"{field_def.label} 不能小于 {field_def.minimum:g}")
        if field_def.maximum is not None and number > field_def.maximum:
            raise ParamInvalidError(f"{field_def.label} 不能大于 {field_def.maximum:g}")
        return number

    text = str(value).strip() if value is not None else ""
    if field_def.type == "select" and field_def.options and text not in field_def.options:
        raise ParamInvalidError(f"{field_def.label} 只能是: {', '.join(field_def.options)}")
    return text


def _validate_consistency() -> None:
    """跨字段的一致性校验.

    单字段校验管不了字段之间的关系, 而很多配置错误恰恰出在关系上
    (例如子块比父块还大, 父子块结构就失去意义). 这类错误不校验的话,
    不会报错, 只会静默产出垃圾结果 —— 那比直接报错更难排查.
    """
    if settings.child_chunk_size >= settings.parent_chunk_size:
        raise ParamInvalidError("子块大小必须小于父块大小")

    if settings.final_top_k > max(settings.vector_top_k, settings.bm25_top_k, 1):
        raise ParamInvalidError("进入 Prompt 的条数不能超过召回条数")

    if settings.chunk_overlap >= settings.child_chunk_size:
        raise ParamInvalidError("子块重叠不能大于等于子块大小")


def _persist() -> None:
    """把当前可编辑字段的值写到运行时配置文件."""
    settings.ensure_dirs()
    payload = {field_def.key: getattr(settings, field_def.key) for field_def in CONFIG_FIELDS}
    try:
        runtime_config_path().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.exception("写入运行时配置失败 | error=%s", exc)
        raise ParamInvalidError(f"配置已生效但保存失败: {exc}") from exc


def _invalidate_caches(changed: set[str]) -> None:
    """配置变了, 对应的单例缓存必须失效.

    这是运行时配置最容易出错的地方: 改了 Key 但客户端还是用旧的,
    表现为"界面上改了却没生效", 用户会以为系统坏了.
    """
    if changed & {"llm_provider", "llm_base_url", "llm_api_key", "llm_model", "llm_timeout"}:
        try:
            from app.services.llm import reset_llm_client  # noqa: PLC0415

            reset_llm_client()
        except ImportError:
            # P2 之前还没有 LLM 模块, 属于预期
            pass

    if changed & {"embedding_device", "embedding_model", "embedding_batch_size"}:
        logger.info("向量模型相关配置已变更, 需要重启服务后生效")

    # 语音 provider / 音色 / Key 换了, 缓存的实例必须丢掉.
    # 尤其 dashscope 的 Key: 它既可能被传进 Recognition, 也可能走全局
    # dashscope.api_key —— 后者是**进程级全局状态**, 不主动改会一直用旧 Key.
    if changed & {
        "speech_asr_provider",
        "speech_tts_provider",
        "dashscope_api_key",
        "speech_asr_model",
        "speech_asr_sample_rate",
        "speech_tts_voice",
        "speech_tts_rate",
    }:
        try:
            from app.services.speech import reset_speech_providers  # noqa: PLC0415

            reset_speech_providers()
        except ImportError:
            pass
