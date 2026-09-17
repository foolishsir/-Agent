"""edge-tts 语音合成.

为什么选它
----------
用微软的在线神经音色, **免费、不需要任何 Key**, 中文自然度明显好于浏览器内置的
SpeechSynthesis(后者机械感很重). 对一个要给别人体验的 demo 项目来说,
"不需要再申请一个 Key 就能有像样的声音"是很实际的优势.

代价是**必须联网**. 所以它在设计上是可以被替换的一档:
真要在完全离线环境里跑, 换回浏览器 SpeechSynthesis 即可(前端已支持).

一个必须做的步骤: 口语化
------------------------
送进 edge-tts 的文本**必须先过 normalizer**, 否则 ``**加粗**`` 会被读成
"星号星号加粗星号星号". 这一步放在 provider 内部而不是调用方,
是为了保证**任何**调用路径都不会漏掉它 —— 放在 API 层的话,
将来多一个调用方就会多一个漏掉的地方.
"""

from __future__ import annotations

import time

from app.core.config import settings
from app.core.exceptions import SpeechError
from app.core.logging import get_logger
from app.services.speech.base import SpeechCapability, SynthesisResult
from app.services.speech.normalizer import to_spoken

logger = get_logger("docmind.speech.edge")


class EdgeTTS:
    """edge-tts 合成器."""

    def __init__(
        self,
        *,
        voice: str | None = None,
        rate: str | None = None,
        max_chars: int | None = None,
    ) -> None:
        self._voice = voice or settings.speech_tts_voice
        self._rate = rate or settings.speech_tts_rate
        self._max_chars = max_chars or settings.speech_max_tts_chars

    @property
    def name(self) -> str:
        return f"edge:{self._voice}"

    @property
    def available(self) -> SpeechCapability:
        # 延迟导入: edge-tts 是可选依赖, 放文件顶部会让没装它的用户
        # 整个服务都起不来 —— 而语音只是一个可选功能.
        try:
            import edge_tts  # noqa: F401, PLC0415
        except ImportError:
            return SpeechCapability(
                provider=self.name,
                available=False,
                reason="未安装 edge-tts, 请运行 pip install edge-tts",
            )
        return SpeechCapability(provider=self.name, available=True)

    async def asynthesize(self, text: str) -> SynthesisResult:
        cap = self.available
        if not cap.available:
            raise SpeechError(cap.reason)

        spoken = to_spoken(text, max_chars=self._max_chars)
        if not spoken:
            raise SpeechError("文本去掉 Markdown 标记后没有可朗读内容（可能整段都是代码块或链接）")

        import edge_tts  # noqa: PLC0415

        started = time.perf_counter()
        audio = bytearray()
        try:
            communicate = edge_tts.Communicate(spoken, self._voice, rate=self._rate)
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio" and chunk.get("data"):
                    audio.extend(chunk["data"])
        except Exception as exc:  # noqa: BLE001 - 网络与音色名错误都从这里出来
            raise SpeechError(
                f"语音合成失败: {exc}。edge-tts 需要联网, 请检查网络或换回浏览器发音人"
            ) from exc

        if not audio:
            raise SpeechError("语音合成返回了空音频")

        cost_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "tts.done | provider=%s voice=%s chars=%d->%d bytes=%d cost_ms=%d",
            self.name,
            self._voice,
            len(text),
            len(spoken),
            len(audio),
            cost_ms,
        )
        return SynthesisResult(
            audio=bytes(audio),
            content_type="audio/mpeg",
            text_chars=len(text),
            cost_ms=cost_ms,
            provider=self.name,
            voice=self._voice,
            spoken_text=spoken,
        )


__all__ = ["EdgeTTS"]
