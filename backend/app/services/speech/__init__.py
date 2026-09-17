"""语音服务入口.

对外暴露 ``get_asr_provider()`` / ``get_tts_provider()`` / ``reset_speech_providers()``,
以及能力自检 ``speech_status()``.

缓存的理由和 LLM / Embedding 一样: 构造有成本(读配置、校验依赖),
但**配置一变必须失效** —— 由 ``config_service._invalidate_caches`` 调用
``reset_speech_providers()``. 漏掉这一步的典型症状是"界面上换了 Key 但没生效".
"""

from __future__ import annotations

import threading

from app.core.config import settings
from app.core.logging import get_logger
from app.services.speech.base import (
    ASRProvider,
    SpeechCapability,
    SynthesisResult,
    Transcription,
    TTSProvider,
)
from app.services.speech.dashscope_asr import DashScopeASR
from app.services.speech.edge_tts_provider import EdgeTTS
from app.services.speech.normalizer import to_spoken

logger = get_logger("docmind.speech")

_lock = threading.Lock()
_asr: ASRProvider | None = None
_tts: TTSProvider | None = None
#: 记录构造时用的配置, 用来发现"配置变了但缓存没清"
_asr_key: tuple[object, ...] | None = None
_tts_key: tuple[object, ...] | None = None


def _build_asr() -> ASRProvider | None:
    provider = (settings.speech_asr_provider or "none").lower()
    if provider == "dashscope":
        return DashScopeASR()
    return None


def _build_tts() -> TTSProvider | None:
    provider = (settings.speech_tts_provider or "none").lower()
    if provider == "edge":
        return EdgeTTS()
    return None


def _asr_signature() -> tuple[object, ...]:
    return (
        settings.speech_asr_provider,
        settings.dashscope_api_key,
        settings.speech_asr_model,
        settings.speech_asr_sample_rate,
    )


def _tts_signature() -> tuple[object, ...]:
    return (
        settings.speech_tts_provider,
        settings.speech_tts_voice,
        settings.speech_tts_rate,
        settings.speech_max_tts_chars,
    )


def get_asr_provider() -> ASRProvider | None:
    """取 ASR 实例; 返回 None 表示语音输入被关闭或未配置."""
    global _asr, _asr_key
    signature = _asr_signature()
    # 签名比对是一层保险: 万一有代码路径忘了调 reset, 这里也能自愈,
    # 不用等到用户发现"改了没生效"再回头查
    if _asr is not None and _asr_key == signature:
        return _asr
    with _lock:
        if _asr is None or _asr_key != signature:
            _asr = _build_asr()
            _asr_key = signature
            logger.info("ASR 提供方已初始化 | %s", _asr.name if _asr else "disabled")
    return _asr


def get_tts_provider() -> TTSProvider | None:
    global _tts, _tts_key
    signature = _tts_signature()
    if _tts is not None and _tts_key == signature:
        return _tts
    with _lock:
        if _tts is None or _tts_key != signature:
            _tts = _build_tts()
            _tts_key = signature
            logger.info("TTS 提供方已初始化 | %s", _tts.name if _tts else "disabled")
    return _tts


def reset_speech_providers() -> None:
    """丢弃缓存, 下次调用按最新配置重建."""
    global _asr, _tts, _asr_key, _tts_key
    with _lock:
        _asr = None
        _tts = None
        _asr_key = None
        _tts_key = None
    logger.info("语音提供方已重置, 下次调用将使用最新配置")


def speech_status() -> dict[str, object]:
    """给前端和健康探针用的能力自检.

    前端需要它来决定**录音按钮要不要禁用**:
    如果 ASR 没配好还让用户按下去, 等他说完 30 秒再报"没配 Key",
    体验比直接禁用差得多.
    """
    asr = get_asr_provider()
    tts = get_tts_provider()

    asr_cap = (
        asr.available
        if asr
        else SpeechCapability(
            provider="none",
            available=False,
            reason="语音输入已关闭（设置 → 语音 → 语音输入选 none）",
        )
    )
    tts_cap = (
        tts.available
        if tts
        else SpeechCapability(
            provider="none",
            available=False,
            reason="语音输出已关闭（设置 → 语音 → 语音输出选 none）",
        )
    )

    return {
        "asr": asr_cap.to_dict(),
        "tts": tts_cap.to_dict(),
        "sample_rate": settings.speech_asr_sample_rate,
        "max_audio_seconds": settings.speech_max_audio_seconds,
        "voice": settings.speech_tts_voice if tts else "",
        "rate": settings.speech_tts_rate if tts else "",
    }


__all__ = [
    "ASRProvider",
    "SpeechCapability",
    "SynthesisResult",
    "TTSProvider",
    "Transcription",
    "get_asr_provider",
    "get_tts_provider",
    "reset_speech_providers",
    "speech_status",
    "to_spoken",
]
