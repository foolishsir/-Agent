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
from tempfile import TemporaryDirectory

from app.core.config import settings
from app.core.exceptions import SpeechError
from app.core.logging import get_logger
from app.services.speech.audio_probe import probe_sample_rate
from app.services.speech.base import SpeechCapability, Transcription

logger = get_logger("docmind.speech.dashscope")

#: 文件后缀 → Paraformer 的容器格式标识.
#:
#: **不能写死成 "wav"**: 接口层允许上传 mp3/m4a/webm 等格式(方便用现成音频调试),
#: 把 mp3 声明成 wav 不会报错, 而是**识别出一堆乱码** —— 这类失败极难归因,
#: 因为接口返回 200、日志也正常.
#:
#: 只列 Paraformer 官方支持的格式; 表里没有的一律退回 wav(主路径就是 wav).
_FORMAT_BY_SUFFIX: dict[str, str] = {
    "wav": "wav",
    "mp3": "mp3",
    "pcm": "pcm",
    "opus": "opus",
    "ogg": "opus",  # 浏览器录音常见的容器, 内容就是 opus
    "speex": "speex",
    "aac": "aac",
    "m4a": "aac",  # m4a 是 aac 的容器
    "amr": "amr",
}


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

        # SDK 的同步接口只接受**文件路径**, 不接受字节流, 所以要用临时文件桥接.
        #
        # 这里**不能用 NamedTemporaryFile**: 它在 Windows 上以独占方式打开文件
        # (CreateFile 不带 FILE_SHARE_READ), 而 SDK 内部会自己 open 这个路径去读,
        # 于是直接 Permission denied:
        #
        #     [Errno 13] Permission denied: 'C:\\...\\Temp\\tmpq5y532d0.wav'
        #
        # POSIX 允许同一路径被多个句柄打开, 所以这个坑**只在 Windows 上出现** ——
        # Linux 上开发永远碰不到, 本地测试也照样绿.
        #
        # 正确做法: 用 TemporaryDirectory 拿一个目录, 手动写文件并**先关闭句柄**,
        # 再把路径交给 SDK. 目录级清理还能顺带处理"SDK 没释放句柄"的残留.
        suffix = Path(filename).suffix or ".wav"
        audio_format = _FORMAT_BY_SUFFIX.get(suffix.lower().lstrip("."), "wav")

        # 采样率**必须取文件里的真实值**, 不能直接用配置值.
        #
        # Paraformer 会校验声明值与文件头是否一致, 不一致直接报:
        #   Failed to decode audio: sample rate 16000 not equals with real 24000
        # 真实撞到过: edge-tts 合成的是 24kHz mp3, 而配置里的 16000 是给
        # 浏览器录音(wav)用的 —— 把配置值无条件传过去就炸了.
        #
        # 读不出来的格式(裸 pcm 没有头)才退回配置值, 这是唯一必须靠声明的场景.
        probed = probe_sample_rate(audio, filename)
        sample_rate = probed or self._sample_rate
        logger.debug(
            "asr.sample_rate | probed=%s configured=%s used=%d",
            probed,
            self._sample_rate,
            sample_rate,
        )

        started = time.perf_counter()
        # ignore_cleanup_errors: SDK 或防火墙软件偶尔会短暂持有句柄,
        # 让它把清理失败咽下去 —— 临时目录留在系统 Temp 里无害,
        # 但为此让一次成功的识别变成报错是不可接受的.
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            path = Path(tmpdir) / f"audio{suffix}"
            path.write_bytes(audio)
            # 句柄已随 write_bytes 关闭, 这里开始 SDK 才能打开它
            sentences = await asyncio.to_thread(
                self._recognize_sync, str(path), audio_format, sample_rate
            )
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
            audio_format=audio_format,
            sample_rate=sample_rate,
        )

    def _recognize_sync(
        self, path: str, audio_format: str = "wav", sample_rate: int | None = None
    ) -> list[str]:
        """真正调用 SDK. 在 worker 线程里跑, 不阻塞事件循环.

        「把阻塞调用丢进线程池」这一点值得单独说: ``Recognition.call`` 内部是
        WebSocket 收发, 单段音频几百毫秒到几秒. 直接在协程里同步调用会把
        整个事件循环卡住 —— 表现是"语音识别期间其他接口全部无响应".

        Args:
            path: 音频文件路径
            audio_format: 容器格式(pcm/wav/mp3/...).
                **必须和文件真实格式一致**: 把 mp3 声明成 wav 不会报错,
                而是识别出一堆乱码 —— 这类失败很难归因.
            sample_rate: 采样率. **必须和文件真实值一致**, 否则 SDK 直接报错.
                默认取配置值, 但调用方通常会传入从文件头读出的真实值.
        """
        # 同样延迟导入(见 available 的说明). 这里还在**工作线程**里,
        # 首次导入的几十毫秒不会卡住事件循环.
        from dashscope.audio.asr import Recognition, RecognitionCallback  # noqa: PLC0415

        # 回调是**构造函数的必填参数**, 但 Recognition.call() 全程不会调它 ——
        # 它自己内部收集分句, 通过返回值给出结果.
        #
        # 这一条是踩出来的: 最初的实现把 on_event 当成主要结果来源,
        # 结果真实调用时永远拿到空字符串(而桩测试全绿, 因为桩不体现 SDK 的真实契约).
        # 教训: **mock 只能验证"我以为的契约", 验证不了契约本身对不对.**
        class _NoopCallback(RecognitionCallback):
            """占位回调. 结果不走这里, 见下方对返回值的解析."""

            def on_open(self) -> None:
                return

            def on_event(self, result) -> None:  # noqa: ARG002 - 协议要求的方法签名
                return

            def on_complete(self) -> None:
                return

            def on_close(self) -> None:
                return

            def on_error(self, result) -> None:  # pragma: no cover - 需要真实网络
                logger.warning("asr.sdk_error | result=%s", result)

        recognition = Recognition(
            model=self._model,
            callback=_NoopCallback(),
            format=audio_format,
            sample_rate=sample_rate or self._sample_rate,
        )
        try:
            result = recognition.call(path, api_key=self._api_key)
        except Exception as exc:  # noqa: BLE001 - SDK 异常类型不稳定
            raise SpeechError(f"语音识别调用失败: {exc}") from exc

        status = getattr(result, "status_code", None)
        if status is not None and status != 200:
            message = getattr(result, "message", "") or "未知错误"
            raise SpeechError(f"语音识别失败({status}): {message}")

        return _extract_texts(result)


def _extract_texts(result: object) -> list[str]:
    """从 SDK 返回值里取分句文本.

    ``get_sentence()`` 的返回类型是 **union**, 这是最容易写错的地方:

    - 有完整分句时 → ``RecognitionResult.__init__`` 把内部 ``sentences`` 列表
      塞进 ``output["sentence"]``, 于是返回 **list[dict]**
    - 只有一个不完整结果时 → 直接用服务端响应里的 ``output``, 于是返回 **dict**

    (依据: ``dashscope/audio/asr/recognition.py`` 的 ``RecognitionResult.__init__``)

    只处理 dict 分支的话, **正常识别出内容时反而拿到空串** ——
    而"返回空"看起来像"用户没说话", 排查方向会完全跑偏.
    """
    try:
        payload = result.get_sentence()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - 结构变化不该让整段识别失败
        logger.exception("解析识别结果失败")
        return []

    if isinstance(payload, list):
        return [
            str(item["text"]) for item in payload if isinstance(item, dict) and item.get("text")
        ]
    if isinstance(payload, dict) and payload.get("text"):
        return [str(payload["text"])]
    return []


__all__ = ["DashScopeASR"]
