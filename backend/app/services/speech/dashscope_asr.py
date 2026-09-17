"""阿里云百炼(DashScope) Paraformer 语音识别.

为什么选它
----------
中文识别质量在开源/商用方案里第一梯队, 且 ``dashscope`` SDK **项目里本来就装了**
(LLM 的 dashscope 服务商选项用它), 不引入新依赖.

用的是**同步识别** ``Recognition.call(file=...)`` 而不是异步文件转写
``Transcription.async_call``:
- 异步接口要先传文件到 OSS 拿 URL, 再轮询任务状态, 短音频上纯属浪费;
- 面试回答只有几十秒, 同步接口刚好.

两个容易踩的坑
--------------
**① 采样率必须两端一致.** ``Recognition(model, callback, format, sample_rate)``
里的 ``sample_rate`` 必须和音频实际采样率相同. 不一致时**不会报错**,
而是识别出一堆乱码 —— 这类问题最难查. 所以前端采集端被固定成 16k 单声道,
这个值也从配置读, 两边同一个来源.

**② ``dashscope.api_key`` 是进程级全局状态.** SDK 会优先读这个全局变量.
如果界面上换了 Key 却不更新它, 识别会一直用旧 Key —— 表现为"改了没生效".
所以这里每次调用都显式传 ``api_key=``, 不依赖全局.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from app.core.config import settings
from app.core.exceptions import SpeechError
from app.core.logging import get_logger
from app.services.speech.base import SpeechCapability, Transcription

logger = get_logger("docmind.speech.dashscope")


@dataclass
class _Collector:
    """把回调里陆续到达的分句收集起来.

    ``Recognition.call`` 是**同步阻塞**的, 但它通过 callback 交付结果.
    这里用一个朴素的收集器把两者桥接起来 —— 不玩花活,
    因为同步调用返回时所有分句都已经到齐了.
    """

    sentences: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sentences is None:
            self.sentences = []

    @property
    def text(self) -> str:
        return "".join(self.sentences).strip()


class DashScopeASR:
    """Paraformer 实时识别(同步模式)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        sample_rate: int | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else settings.dashscope_api_key
        self._model = model or settings.speech_asr_model
        self._sample_rate = sample_rate or settings.speech_asr_sample_rate

    @property
    def name(self) -> str:
        return f"dashscope:{self._model}"

    @property
    def available(self) -> SpeechCapability:
        if not self._api_key:
            return SpeechCapability(
                provider=self.name,
                available=False,
                reason="未配置阿里云百炼 Key（设置 → 语音 → 阿里云百炼 Key）",
            )
        # 延迟导入而不是放文件顶部: dashscope 是**可选依赖** ——
        # 不用语音功能的用户不该因为它没装就起不来服务.
        # 放顶部的话 ImportError 会在模块加载时抛出, 直接把整个服务带崩.
        try:
            import dashscope  # noqa: F401, PLC0415
        except ImportError:
            return SpeechCapability(
                provider=self.name,
                available=False,
                reason="未安装 dashscope SDK, 请运行 pip install dashscope",
            )
        return SpeechCapability(provider=self.name, available=True)

    async def atranscribe(self, audio: bytes, *, filename: str = "audio.wav") -> Transcription:
        cap = self.available
        if not cap.available:
            raise SpeechError(cap.reason)

        if not audio:
            raise SpeechError("音频内容为空")

        # SDK 的同步接口只接受**文件路径**, 不接受字节流.
        # 用一个临时文件桥接 —— 注意必须带正确的后缀, SDK 靠它判断容器格式.
        suffix = Path(filename).suffix or ".wav"
        started = time.perf_counter()
        with NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio)
            tmp.flush()
            sentences = await asyncio.to_thread(self._recognize_sync, tmp.name)
        cost_ms = int((time.perf_counter() - started) * 1000)

        text = "".join(sentences).strip()
        logger.info(
            "asr.done | provider=%s chars=%d sentences=%d cost_ms=%d",
            self.name,
            len(text),
            len(sentences),
            cost_ms,
        )
        return Transcription(
            text=text,
            cost_ms=cost_ms,
            sentences=sentences,
            provider=self.name,
        )

    def _recognize_sync(self, path: str) -> list[str]:
        """真正调用 SDK. 在 worker 线程里跑, 不阻塞事件循环.

        「把阻塞调用丢进线程池」这一点值得单独说: ``Recognition.call`` 内部是
        WebSocket 收发, 单段音频几百毫秒到几秒. 直接在协程里同步调用会把
        整个事件循环卡住 —— 表现是"语音识别期间其他接口全部无响应".
        """
        # 同样延迟导入(见 available 的说明). 这里还在**工作线程**里,
        # 首次导入的几十毫秒不会卡住事件循环.
        from dashscope.audio.asr import Recognition, RecognitionCallback  # noqa: PLC0415

        collected: list[str] = []

        class _CB(RecognitionCallback):
            def on_event(self, result) -> None:
                try:
                    sentence = result.get_sentence()
                except Exception:  # noqa: BLE001 - 结构变化不该让整段识别失败
                    return
                if not sentence:
                    return
                if isinstance(sentence, dict):
                    piece = sentence.get("text", "")
                    if result.is_sentence_end(sentence) and piece:
                        collected.append(piece)
                elif isinstance(sentence, list):
                    for item in sentence:
                        if isinstance(item, dict) and item.get("text"):
                            collected.append(item["text"])

            def on_error(self, result) -> None:  # pragma: no cover - 需要真实网络
                logger.warning("asr.callback_error | result=%s", result)

            def on_complete(self) -> None:
                return

            def on_close(self) -> None:
                return

        recognition = Recognition(
            model=self._model,
            callback=_CB(),
            format="wav",
            sample_rate=self._sample_rate,
        )
        try:
            result = recognition.call(path, api_key=self._api_key)
        except Exception as exc:  # noqa: BLE001 - SDK 异常类型不稳定
            raise SpeechError(f"语音识别调用失败: {exc}") from exc

        status = getattr(result, "status_code", None)
        if status is not None and status != 200:
            message = getattr(result, "message", "") or "未知错误"
            raise SpeechError(f"语音识别失败({status}): {message}")

        # 回调偶尔比 call() 返回晚一拍, 兜底从 result 里再补一次
        if not collected:
            try:
                sentence = result.get_sentence()
                if isinstance(sentence, dict) and sentence.get("text"):
                    collected.append(sentence["text"])
            except Exception:  # noqa: BLE001
                pass
        return collected


__all__ = ["DashScopeASR"]
