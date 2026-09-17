"""语音接口.

    GET  /speech/status      能力自检(前端据此决定录音按钮是否可用)
    POST /speech/transcribe  音频 → 文字
    POST /speech/synthesize  文字 → 音频（直接返回音频字节, 不套统一的 JSON 信封）
    POST /speech/spoken-text 只做口语化转换, 不合成 —— 用来排查"为什么读出来是这样"

**为什么 /synthesize 不套统一响应格式**: 它是二进制流媒体, 不是数据接口.
塞进 `{"code":"OK","data":{"audio":"<base64>"}}` 会让音频体积膨胀 33%,
前端还得先解码再喂给 <audio>. 直接返回 audio/mpeg 更省事也更快.

**音频格式约定**: 前端统一上传 **16kHz 单声道 WAV**.
不用 MediaRecorder 的 WebM/Opus, 是因为后端解 WebM 需要 ffmpeg ——
一个 70MB 的外部二进制. 而浏览器本来就能直接给出 16kHz PCM,
在采集端重采样比在服务端装解码器干净得多.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Response, UploadFile
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.exceptions import ParamInvalidError, SpeechError
from app.core.logging import get_logger, log_kv
from app.core.response import ok
from app.services.speech import get_asr_provider, get_tts_provider, speech_status, to_spoken

router = APIRouter()
logger = get_logger("docmind.api.speech")

#: 允许的音频容器后缀. WAV 是主路径, 其余留给"用现成音频文件调试"的场景.
_ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".opus", ".pcm", ".flac"}


@router.get("/status", summary="语音能力自检")
async def get_status() -> dict[str, object]:
    """返回 ASR / TTS 当前可不可用, 以及不可用的原因.

    前端拿它做两件事:
    1. ASR 不可用 → 录音按钮置灰, 并把 ``reason`` 显示成 tooltip
    2. 显示"16kHz / 最长 180 秒"这类约束, 让用户知道边界在哪
    """
    return ok(speech_status())


@router.post("/transcribe", summary="语音识别：音频转文字")
async def transcribe(file: UploadFile = File(...)) -> dict[str, object]:
    """上传一段音频, 返回识别文本.

    走 multipart 而不是 base64 JSON: 音频是二进制, base64 会白白膨胀 33%,
    而且浏览器 ``FormData`` 直接就能发 ``Blob``, 前端代码更短.
    """
    provider = get_asr_provider()
    if provider is None:
        raise SpeechError("语音输入未启用, 请在「设置 → 语音」里选择 ASR 提供方")

    capability = provider.available
    if not capability.available:
        raise SpeechError(capability.reason)

    audio = await file.read()
    if not audio:
        raise ParamInvalidError("上传的音频为空")

    # 体积上限: 16kHz 单声道 16bit 每秒 32KB, 180 秒约 5.5MB.
    # 给 2 倍余量, 挡住"误传了一个几十 MB 的文件"这种输入.
    max_bytes = settings.speech_max_audio_seconds * 32_000 * 2
    if len(audio) > max_bytes:
        raise ParamInvalidError(
            f"音频过大({len(audio) / 1024 / 1024:.1f}MB), "
            f"按 {settings.speech_max_audio_seconds} 秒上限最多约 {max_bytes / 1024 / 1024:.1f}MB"
        )

    filename = file.filename or "audio.wav"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ".wav"
    if suffix not in _ALLOWED_SUFFIXES:
        raise ParamInvalidError(
            f"不支持的音频格式 {suffix}, 支持: {', '.join(sorted(_ALLOWED_SUFFIXES))}"
        )

    result = await provider.atranscribe(audio, filename=filename)
    log_kv(
        logger,
        "speech.transcribed",
        provider=provider.name,
        in_bytes=len(audio),
        out_chars=len(result.text),
        cost_ms=result.cost_ms,
    )

    return ok(
        {
            **result.to_dict(),
            "empty": not result.text,
        }
    )


class SynthesizeRequest(BaseModel):
    """合成请求.

    ``text`` 有长度上限, 但它**不是**主要的防护手段 ——
    真正的防护是 ``settings.speech_max_tts_chars``, 在 normalizer 里做句子级截断.
    这里的 Field 限制只是为了让请求在进入业务逻辑之前就被 Pydantic 挡掉,
    省一次无谓的语音服务调用.
    """

    text: str = Field(min_length=1, max_length=5000)


@router.post("/synthesize", summary="语音合成：文字转音频")
async def synthesize(payload: SynthesizeRequest) -> Response:
    """把文本合成成音频并直接返回音频字节.

    返回体是 ``audio/mpeg``, 不是 JSON. 前端拿 ``response.blob()``
    直接塞进 ``<audio>`` 或 ``Audio()``.
    """
    provider = get_tts_provider()
    if provider is None:
        raise SpeechError("语音输出未启用, 请在「设置 → 语音」里选择 TTS 提供方")

    capability = provider.available
    if not capability.available:
        raise SpeechError(capability.reason)

    result = await provider.asynthesize(payload.text)
    log_kv(
        logger,
        "speech.synthesized",
        provider=provider.name,
        in_chars=result.text_chars,
        spoken_chars=len(result.spoken_text),
        out_bytes=len(result.audio),
        cost_ms=result.cost_ms,
    )

    return Response(
        content=result.audio,
        media_type=result.content_type,
        headers={
            "Cache-Control": "no-store",
            # 把口语化后的文本回传, 排查读音问题时第一步就是看它.
            # 放在响应头而不是 JSON 里, 是为了保持响应体是纯音频.
            "X-Spoken-Chars": str(len(result.spoken_text)),
            "X-Synth-Cost-Ms": str(result.cost_ms),
        },
    )


class SpokenTextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@router.post("/spoken-text", summary="只看口语化转换结果（不合成）")
async def spoken_text(payload: SpokenTextRequest) -> dict[str, object]:
    """调试用: 看一段文本会被朗读成什么样.

    单独做成接口是因为**这是排查读音问题唯一的入手点**.
    用户反馈"读得不对"时, 先看这一版文本, 就能区分是
    "标记没清干净"还是"音色本身读错了" —— 后者再怎么调文本也没用.
    """
    spoken = to_spoken(payload.text, max_chars=settings.speech_max_tts_chars)
    return ok(
        {
            "original": payload.text,
            "spoken": spoken,
            "original_chars": len(payload.text),
            "spoken_chars": len(spoken),
            "empty": not spoken,
        }
    )
