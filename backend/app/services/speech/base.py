"""语音服务的协议与数据结构.

为什么先定义协议
----------------
和 ``embedding/`` ``llm/`` ``skills/`` 一样, 这里也是**先定接口再定实现**:

- ASR 和 TTS 都是"外部能力依赖", 供应商更换频繁(今天阿里, 明天本地模型);
- 测试里必须能塞桩实现, 否则单测要么联网要么被跳过 —— 两个都不能接受.

接口刻意定得很窄: 一个方法、一个数据类.
窄接口的价值在于**换实现时不用改调用方** —— 调用方只知道自己给了音频、
拿回文字, 不关心背后是 Paraformer 还是 Whisper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Transcription:
    """一次语音识别的结果."""

    text: str
    #: 音频时长(毫秒), 用于在前端显示"识别了 12.3 秒"
    duration_ms: int = 0
    #: 识别耗时(毫秒). 和 duration_ms 一起看能算出实时率(RTF),
    #: 这个指标在做流式优化时是基线
    cost_ms: int = 0
    #: 分句结果. Paraformer 会给句级时间戳, 保留下来便于调试
    sentences: list[str] = field(default_factory=list)
    provider: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "duration_ms": self.duration_ms,
            "cost_ms": self.cost_ms,
            "sentences": self.sentences,
            "provider": self.provider,
        }


@dataclass
class SynthesisResult:
    """一次语音合成的结果."""

    audio: bytes
    content_type: str = "audio/mpeg"
    #: 送进合成器的**原始文本长度**(不是口语化之后的), 用于计费/限流口径一致
    text_chars: int = 0
    cost_ms: int = 0
    provider: str = ""
    voice: str = ""
    #: 口语化之后的文本. 返回给前端是为了让"为什么读出来是这样"可排查 ——
    #: 出现读音问题时, 第一步就是看这版文本
    spoken_text: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "content_type": self.content_type,
            "text_chars": self.text_chars,
            "cost_ms": self.cost_ms,
            "provider": self.provider,
            "voice": self.voice,
            "spoken_text": self.spoken_text,
            "audio_bytes": len(self.audio),
        }


@dataclass
class SpeechCapability:
    """某个 provider 当前能不能用, 以及为什么不能用."""

    provider: str
    available: bool
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"provider": self.provider, "available": self.available, "reason": self.reason}


# --------------------------------------------------------------------------- #
# 协议
# --------------------------------------------------------------------------- #
@runtime_checkable
class ASRProvider(Protocol):
    """语音识别提供方."""

    @property
    def name(self) -> str: ...

    @property
    def available(self) -> SpeechCapability: ...

    async def atranscribe(self, audio: bytes, *, filename: str = "audio.wav") -> Transcription:
        """把一段音频转成文字.

        设计成"整段"而不是"流式", 是刻意的:
        面试回答是**说完再评**的场景, 不需要边说边出字.
        流式识别的复杂度(增量解码 + 部分结果合并)在这个场景换不来体验提升.
        """
        ...


@runtime_checkable
class TTSProvider(Protocol):
    """语音合成提供方."""

    @property
    def name(self) -> str: ...

    @property
    def available(self) -> SpeechCapability: ...

    async def asynthesize(self, text: str) -> SynthesisResult:
        """把文本合成成音频."""
        ...


__all__ = [
    "ASRProvider",
    "SpeechCapability",
    "SynthesisResult",
    "TTSProvider",
    "Transcription",
]
