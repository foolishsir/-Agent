"""语音接口的集成测试（桩 provider，不联网）。

为什么用桩而不是真调
--------------------
真调会同时引入两个不稳定因素: **网络**和**额度**.
测试一旦依赖外部服务, 结果就不可复现 —— 别人 clone 下来跑 CI 会因为
"没有百炼 Key"而红, 那是测试设计的问题, 不是代码的问题.

真实链路(edge-tts 合成 + Paraformer 识别)由手工验证覆盖,
``backend/scripts/smoke_speech.py`` 就是那个脚本.

这里测的是**装配层**:
- 音频校验(空/超大/格式)有没有真的拦住
- provider 不可用时错误码对不对(503 而不是 500)
- 合成响应是不是真的二进制音频(而不是被 JSON 信封包了一层)
- 口语化有没有在 provider 内部生效(而不是漏在某条调用路径上)
"""

from __future__ import annotations

import io
from typing import Any

import pytest

from app.services.speech.base import (
    SpeechCapability,
    SynthesisResult,
    Transcription,
)


# --------------------------------------------------------------------------- #
# 桩 provider
# --------------------------------------------------------------------------- #
class StubASR:
    """假的识别器: 原样返回一段固定文本, 并记录收到的音频."""

    name = "stub:asr"

    def __init__(
        self,
        *,
        text: str = "我用了 Redis 缓存热点订单数据。",
        available: bool = True,
        reason: str = "",
    ) -> None:
        self._text = text
        self._available = available
        self._reason = reason
        self.received: list[bytes] = []
        self.filenames: list[str] = []

    @property
    def available(self) -> SpeechCapability:
        return SpeechCapability(provider=self.name, available=self._available, reason=self._reason)

    async def atranscribe(self, audio: bytes, *, filename: str = "audio.wav") -> Transcription:
        from app.core.exceptions import SpeechError

        if not self._available:
            raise SpeechError(self._reason)
        self.received.append(audio)
        self.filenames.append(filename)
        return Transcription(
            text=self._text,
            cost_ms=42,
            sentences=[self._text] if self._text else [],
            provider=self.name,
        )


class StubTTS:
    """假的合成器: 返回一段固定字节, 但**真实走一遍 normalizer**。

    这点很重要 —— 口语化是在 provider 内部做的, 桩如果跳过它,
    "标记有没有被清干净"这条契约就测不到了.
    """

    name = "stub:tts"

    def __init__(self, *, available: bool = True, reason: str = "") -> None:
        self._available = available
        self._reason = reason
        self.spoken_seen: list[str] = []

    @property
    def available(self) -> SpeechCapability:
        return SpeechCapability(provider=self.name, available=self._available, reason=self._reason)

    async def asynthesize(self, text: str) -> SynthesisResult:
        from app.core.exceptions import SpeechError
        from app.services.speech.normalizer import to_spoken

        if not self._available:
            raise SpeechError(self._reason)
        spoken = to_spoken(text)
        if not spoken:
            raise SpeechError("去掉 Markdown 标记后没有可朗读内容")
        self.spoken_seen.append(spoken)
        return SynthesisResult(
            audio=b"ID3fake-mp3-bytes",
            content_type="audio/mpeg",
            text_chars=len(text),
            cost_ms=17,
            provider=self.name,
            voice="stub-voice",
            spoken_text=spoken,
        )


@pytest.fixture
def stub_asr(monkeypatch: pytest.MonkeyPatch) -> StubASR:
    from app.api.v1 import speech as speech_api

    stub = StubASR()
    monkeypatch.setattr(speech_api, "get_asr_provider", lambda: stub)
    return stub


@pytest.fixture
def stub_tts(monkeypatch: pytest.MonkeyPatch) -> StubTTS:
    from app.api.v1 import speech as speech_api

    stub = StubTTS()
    monkeypatch.setattr(speech_api, "get_tts_provider", lambda: stub)
    return stub


# --------------------------------------------------------------------------- #
# /speech/status
# --------------------------------------------------------------------------- #
def test_status_reports_capability_reasons(client):
    """能力自检必须给出**不可用的原因**。

    前端拿它做两件事: 置灰录音按钮, 以及把原因显示成 tooltip。
    只返回 available=false 而不说为什么, 用户只能干瞪眼。
    """
    data = client.get("/api/v1/speech/status").json()["data"]
    for key in ("asr", "tts"):
        assert key in data
        assert "available" in data[key]
        assert "provider" in data[key]
        if not data[key]["available"]:
            assert data[key]["reason"], f"{key} 不可用却没给原因"
    assert data["sample_rate"] == 16000
    assert data["max_audio_seconds"] > 0


# --------------------------------------------------------------------------- #
# /speech/transcribe
# --------------------------------------------------------------------------- #
def test_transcribe_returns_text(client, stub_asr):
    res = client.post(
        "/api/v1/speech/transcribe",
        files={"file": ("answer.wav", b"RIFFfake-wav-bytes", "audio/wav")},
    )
    assert res.status_code == 200, res.text
    data = res.json()["data"]

    assert data["text"].startswith("我用了 Redis")
    assert data["empty"] is False
    assert data["provider"] == "stub:asr"
    assert data["cost_ms"] == 42
    # 音频必须**原样**传给 provider —— 中间被转码或截断的话, 识别结果就不可信了
    assert stub_asr.received == [b"RIFFfake-wav-bytes"]
    assert stub_asr.filenames == ["answer.wav"]


def test_transcribe_reports_detected_format_and_sample_rate(client, monkeypatch):
    """响应里要带服务端**实际判定**的格式与采样率.

    这两个值是排查"识别不对"的入手点: 格式判断错、采样率声明错,
    都表现为"乱码或空结果", 而翻日志才知道服务端当时认为在处理什么太慢。
    """
    from app.api.v1 import speech as speech_api

    class _Provider:
        name = "stub:probe"

        @property
        def available(self):
            return SpeechCapability(provider=self.name, available=True)

        async def atranscribe(self, audio: bytes, *, filename: str = "audio.wav"):
            return Transcription(
                text="结果",
                provider=self.name,
                audio_format="wav",
                sample_rate=16000,
            )

    monkeypatch.setattr(speech_api, "get_asr_provider", lambda: _Provider())
    data = client.post(
        "/api/v1/speech/transcribe",
        files={"file": ("a.wav", b"x" * 100, "audio/wav")},
    ).json()["data"]

    assert data["audio_format"] == "wav"
    assert data["sample_rate"] == 16000


def test_transcribe_marks_empty_result(client, monkeypatch):
    """识别结果为空不是错误(用户可能没说话), 用 empty 标记让前端提示重说。

    返回 500 的话前端只能显示一个吓人的报错, 而实际上什么都没坏。
    """
    from app.api.v1 import speech as speech_api

    stub = StubASR(text="")
    monkeypatch.setattr(speech_api, "get_asr_provider", lambda: stub)

    data = client.post(
        "/api/v1/speech/transcribe",
        files={"file": ("a.wav", b"x" * 100, "audio/wav")},
    ).json()["data"]
    assert data["empty"] is True
    assert data["text"] == ""


def test_transcribe_rejects_empty_audio(client, stub_asr):
    res = client.post(
        "/api/v1/speech/transcribe",
        files={"file": ("a.wav", b"", "audio/wav")},
    )
    assert res.status_code == 400
    assert "为空" in res.json()["message"]


def test_transcribe_rejects_unsupported_format(client, stub_asr):
    res = client.post(
        "/api/v1/speech/transcribe",
        files={"file": ("a.txt", b"not audio", "text/plain")},
    )
    assert res.status_code == 400
    assert "不支持" in res.json()["message"]


def test_transcribe_rejects_oversized_audio(client, stub_asr):
    """体积上限的作用是挡住"误传了一个几十 MB 的文件", 而不是限制正常录音。"""
    from app.core.config import settings

    too_big = b"x" * (settings.speech_max_audio_seconds * 32_000 * 2 + 1024)
    res = client.post(
        "/api/v1/speech/transcribe",
        files={"file": ("a.wav", too_big, "audio/wav")},
    )
    assert res.status_code == 400
    assert "过大" in res.json()["message"]
    # 超限的文件不该被送进识别接口 —— 那是白花钱
    assert stub_asr.received == []


def test_transcribe_unavailable_provider_is_503(client, monkeypatch):
    """provider 没配好 → 503 而不是 500。

    503 的语义是"服务端缺配置, 重试无用"; 500 会让调用方以为可以重试。
    """
    from app.api.v1 import speech as speech_api

    stub = StubASR(available=False, reason="未配置阿里云百炼 Key")
    monkeypatch.setattr(speech_api, "get_asr_provider", lambda: stub)

    res = client.post(
        "/api/v1/speech/transcribe",
        files={"file": ("a.wav", b"x" * 100, "audio/wav")},
    )
    assert res.status_code == 502
    assert "阿里云百炼" in res.json()["message"]


def test_transcribe_when_provider_disabled(client, monkeypatch):
    """provider 为 None(功能被关掉)→ 提示去设置里启用, 而不是崩溃。"""
    from app.api.v1 import speech as speech_api

    monkeypatch.setattr(speech_api, "get_asr_provider", lambda: None)
    res = client.post(
        "/api/v1/speech/transcribe",
        files={"file": ("a.wav", b"x" * 100, "audio/wav")},
    )
    assert res.status_code == 502
    assert "未启用" in res.json()["message"]


# --------------------------------------------------------------------------- #
# /speech/synthesize
# --------------------------------------------------------------------------- #
def test_synthesize_returns_raw_audio_not_json(client, stub_tts):
    """响应体必须是**纯音频字节**, 不能被统一的 JSON 信封包一层。

    包一层的话音频体积膨胀 33%(base64), 前端还得先解码再喂给 <audio>。
    """
    res = client.post("/api/v1/speech/synthesize", json={"text": "你为什么用 Redis？"})

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("audio/mpeg")
    assert res.content == b"ID3fake-mp3-bytes"
    # 确认真的是二进制而不是 {"code":"OK",...}
    assert not res.content.startswith(b"{")


def test_synthesize_normalizes_markdown_before_speaking(client, stub_tts):
    """口语化必须生效 —— 否则会读出"星号星号为什么星号星号"。"""
    client.post(
        "/api/v1/speech/synthesize",
        json={"text": "**为什么**用 `Chroma`？见[文档](https://x.com/a) [1]"},
    )
    spoken = stub_tts.spoken_seen[-1]
    assert "**" not in spoken
    assert "`" not in spoken
    assert "http" not in spoken
    assert "[1]" not in spoken
    assert "为什么" in spoken and "Chroma" in spoken


def test_synthesize_exposes_cost_in_headers(client, stub_tts):
    """耗时放响应头 —— 响应体要留给音频, 没地方放 JSON 元信息。"""
    res = client.post("/api/v1/speech/synthesize", json={"text": "测试"})
    assert res.headers["x-synth-cost-ms"] == "17"


def test_synthesize_rejects_empty_text(client, stub_tts):
    """空文本在 Pydantic 层就被挡掉, 省一次无谓的语音服务调用。

    状态码是 400 而不是 FastAPI 默认的 422 —— 项目里全局把校验错误
    统一成了 400, 保持和其他接口一致(错误体也是统一的 code/message 信封)。
    """
    res = client.post("/api/v1/speech/synthesize", json={"text": ""})
    assert res.status_code == 400
    assert res.json()["code"] == "PARAM_INVALID"
    assert stub_tts.spoken_seen == []


def test_synthesize_unavailable_provider(client, monkeypatch):
    from app.api.v1 import speech as speech_api

    stub = StubTTS(available=False, reason="未安装 edge-tts, 请运行 pip install edge-tts")
    monkeypatch.setattr(speech_api, "get_tts_provider", lambda: stub)

    res = client.post("/api/v1/speech/synthesize", json={"text": "测试"})
    assert res.status_code == 502
    assert "edge-tts" in res.json()["message"]


def test_synthesize_code_only_text_fails_loudly(client, stub_tts):
    """全是代码块的文本清完什么都不剩, 必须**报错而不是返回静音**。

    返回一段空音频的话, 用户会以为程序卡住了 —— 那比明确报错难排查得多。
    """
    res = client.post("/api/v1/speech/synthesize", json={"text": "```python\nprint(1)\n```"})
    assert res.status_code == 502
    assert "可朗读内容" in res.json()["message"]


# --------------------------------------------------------------------------- #
# /speech/spoken-text
# --------------------------------------------------------------------------- #
def test_spoken_text_endpoint_for_debugging(client):
    """排查读音问题的唯一入手点: 先看这版文本, 再决定是换音色还是改规则。"""
    res = client.post(
        "/api/v1/speech/spoken-text",
        json={"text": "# 标题\n**为什么**用 [1] Redis？"},
    )
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["spoken"] == "标题\n为什么用 Redis？"
    assert data["original_chars"] > data["spoken_chars"]
    assert data["empty"] is False


def test_spoken_text_reports_empty_for_code_only(client):
    data = client.post("/api/v1/speech/spoken-text", json={"text": "```\ncode\n```"}).json()["data"]
    assert data["empty"] is True
    assert data["spoken"] == ""


# --------------------------------------------------------------------------- #
# 配置契约
# --------------------------------------------------------------------------- #
def test_speech_settings_exposed_to_frontend(client):
    """语音配置必须在设置接口里出现, 否则界面上配不了。

    这条容易漏: 后端加了 settings 字段, 但忘了加进 CONFIG_FIELDS,
    结果是"配置项存在、能被读、但界面上看不到也改不了"。
    """
    groups = client.get("/api/v1/settings").json()["data"]["groups"]
    names = {g["name"] for g in groups}
    assert "语音" in names, f"设置里没有语音分组, 现有: {names}"

    speech_group = next(g for g in groups if g["name"] == "语音")
    keys = {f["key"] for f in speech_group["fields"]}
    assert {
        "speech_asr_provider",
        "speech_tts_provider",
        "dashscope_api_key",
        "speech_tts_voice",
        "speech_tts_rate",
    } <= keys

    # 两个 Key 必须分开 —— 一个是对话模型(DeepSeek), 一个是语音(百炼).
    # 混在一起的话用户会以为填一个就够.
    llm_group = next(g for g in groups if g["name"] == "大模型")
    assert "llm_api_key" in {f["key"] for f in llm_group["fields"]}
    assert "llm_api_key" not in keys

    # secret 字段不能回传明文
    key_field = next(f for f in speech_group["fields"] if f["key"] == "dashscope_api_key")
    assert key_field["type"] == "secret"


def test_speech_settings_update_resets_providers(client, monkeypatch):
    """改语音配置后必须重置 provider 缓存, 否则"界面上改了没生效"。

    这和 LLM Key 是同一类坑: 单例缓存不失效, 新的 Key 永远用不上。
    """
    from app.services import speech as speech_service

    calls: list[int] = []
    monkeypatch.setattr(speech_service, "reset_speech_providers", lambda: calls.append(1))

    res = client.put(
        "/api/v1/settings",
        json={"speech_tts_voice": "zh-CN-YunxiNeural"},
    )
    assert res.status_code == 200, res.text
    assert calls, "改了语音配置却没有重置 provider 缓存"

    # 还原, 避免影响其他用例
    client.put("/api/v1/settings", json={"speech_tts_voice": "zh-CN-XiaoxiaoNeural"})


def test_transcribe_rejects_unknown_provider_missing_key(monkeypatch):
    """provider 自己报不可用时, 原因要能透传到接口层。

    这是唯一能测到 "未配置 Key" 这条真实路径的地方 ——
    集成测试里 provider 都被换成桩了, 真实 provider 的 available 判断
    只能直接测。
    """
    from app.services.speech.dashscope_asr import DashScopeASR

    provider = DashScopeASR(api_key="")
    cap: dict[str, Any] = provider.available.to_dict()
    assert cap["available"] is False
    assert "百炼" in str(cap["reason"])


def test_edge_tts_provider_reports_missing_dependency(monkeypatch):
    """edge-tts 没装时, available 要给出安装命令, 而不是抛异常。"""
    import builtins

    from app.services.speech.edge_tts_provider import EdgeTTS

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "edge_tts":
            raise ImportError("No module named 'edge_tts'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cap = EdgeTTS().available.to_dict()
    assert cap["available"] is False
    assert "pip install edge-tts" in str(cap["reason"])


def test_wav_upload_roundtrip_shape(client, stub_asr):
    """模拟前端真实发的 multipart 表单, 确认字段名和文件名都被正确接住。"""
    wav = io.BytesIO(b"RIFF" + b"\x00" * 40 + b"data" + b"\x00" * 200)
    res = client.post(
        "/api/v1/speech/transcribe",
        files={"file": ("answer.wav", wav, "audio/wav")},
        headers={"X-User-Id": "speech-test"},
    )
    assert res.status_code == 200
    assert stub_asr.filenames[-1] == "answer.wav"


# --------------------------------------------------------------------------- #
# 临时音频文件的句柄（真实踩到的 Windows 坑）
#
# 用户报错原文:
#   语音识别失败: 语音识别调用失败:
#   [Errno 13] Permission denied: 'C:\...\Temp\tmpq5y532d0.wav'
#
# 根因: 落盘音频时用了 NamedTemporaryFile, 但**它在 Windows 上是独占打开**
# (CreateFile 不带 FILE_SHARE_READ). 而 SDK 内部会自己 open 这个路径去读音频,
# 于是被系统拒绝. POSIX 允许同一路径被多个句柄打开, 所以这个 bug
# **只在 Windows 上出现** —— 在 Linux 上开发永远碰不到, 测试也照样绿.
#
# 修法: TemporaryDirectory + 手动写文件, 先把句柄关掉再把路径交给 SDK.
# --------------------------------------------------------------------------- #
def _transcribe_with_inspecting_sdk(monkeypatch, audio: bytes) -> dict[str, Any]:
    """用一个"自己会打开文件"的假 SDK 跑一遍, 记录它在调用时看到了什么.

    关键点是假 SDK **用独立句柄重新读一次文件** —— 这正是真实 SDK 的行为,
    也正是原 bug 的触发条件.
    """
    import asyncio
    from pathlib import Path

    from app.services.speech.dashscope_asr import DashScopeASR

    rec: dict[str, Any] = {}

    def fake_recognize_sync(
        path: str, audio_format: str = "wav", sample_rate: int | None = None
    ) -> list[str]:
        p = Path(path)
        rec["path"] = path
        rec["format_seen"] = audio_format
        rec["sample_rate_seen"] = sample_rate
        rec["exists_at_call"] = p.exists()
        rec["suffix"] = p.suffix
        rec["parent"] = str(p.parent)
        try:
            # 如果调用方还攥着句柄, Windows 上这里会抛 PermissionError
            rec["content"] = p.read_bytes()
            rec["reopen_ok"] = True
        except OSError as exc:
            rec["reopen_ok"] = False
            rec["reopen_error"] = f"{type(exc).__name__}: {exc}"
            raise
        return ["识别结果"]

    provider = DashScopeASR(api_key="sk-fake-key")
    monkeypatch.setattr(provider, "_recognize_sync", fake_recognize_sync)
    asyncio.run(provider.atranscribe(audio, filename="answer.wav"))
    return rec


def test_audio_file_handle_is_released_before_sdk_reads_it(monkeypatch):
    """落盘后、交给 SDK 前, **调用方必须已经释放文件句柄**.

    **平台差异必须说清楚**: 在 Linux 上, 即使实现退回成"攥着句柄"(旧 bug),
    独立句柄也能正常打开同一个文件, 所以这条断言在 Linux 上**会通过**.
    它真正起作用的地方是 **Windows**(CI 里有 windows-latest runner).
    不要因为本地 Linux 绿了就以为这条防线是摆设 —— 用户就是在 Windows 上中的招.
    """
    rec = _transcribe_with_inspecting_sdk(monkeypatch, b"RIFF-fake-wav-payload")

    assert rec["exists_at_call"] is True, "交给 SDK 时音频文件不存在"
    assert rec["reopen_ok"] is True, f"SDK 打不开音频文件: {rec.get('reopen_error')}"
    assert rec["content"] == b"RIFF-fake-wav-payload"
    # 后缀要保留 —— SDK 靠它判断容器格式, 丢了会当成未知格式
    assert rec["suffix"] == ".wav"


def test_temp_audio_directory_is_cleaned_up(monkeypatch):
    """识别结束后临时文件要清掉, 不能在系统 Temp 里越堆越多.

    这条**在 Windows 上才有意义**: 如果句柄没释放, 连删除都会失败,
    临时文件就会一直堆着. 所以它同时也是句柄泄漏的间接探针.
    """
    from pathlib import Path

    rec = _transcribe_with_inspecting_sdk(monkeypatch, b"RIFF-x")
    leftover = Path(rec["parent"])
    if leftover.exists():
        assert not list(leftover.glob("*.wav")), "临时音频目录没有被清理"


def test_original_suffix_is_passed_through(monkeypatch):
    """输入后缀要透传给落盘文件名和 format 声明, 不能被固定成 wav。

    `.pcm` 是裸流、`.wav` 带头部, SDK 解析方式不同 —— 后缀丢了识别会出错。
    """
    import asyncio

    from app.services.speech.dashscope_asr import DashScopeASR

    seen: dict[str, str] = {}

    def fake_recognize_sync(
        path: str, audio_format: str = "wav", sample_rate: int | None = None
    ) -> list[str]:
        seen["suffix"] = path[path.rfind(".") :]
        seen["format"] = audio_format
        return ["ok"]

    provider = DashScopeASR(api_key="sk-fake-key")
    monkeypatch.setattr(provider, "_recognize_sync", fake_recognize_sync)
    asyncio.run(provider.atranscribe(b"\x00" * 32, filename="recording.mp3"))
    assert seen["suffix"] == ".mp3"
    # format 必须跟着后缀走 —— 把 mp3 声明成 wav 不报错, 只会出乱码
    assert seen["format"] == "mp3"


def test_declared_sample_rate_comes_from_the_file_not_the_config(monkeypatch):
    """采样率必须取**文件头里的真实值**, 不能直接用配置值。

    真实踩到的报错:
        语音识别失败(44): Failed to decode audio:
        sample rate 16000 not equals with real 24000

    起因是 edge-tts 合成的是 24kHz mp3, 而配置里的 16000 是给浏览器录音(wav)用的。
    """
    import asyncio
    import struct

    from app.services.speech.dashscope_asr import DashScopeASR

    seen: dict[str, Any] = {}

    def fake_recognize_sync(
        path: str, audio_format: str = "wav", sample_rate: int | None = None
    ) -> list[str]:
        seen["sample_rate"] = sample_rate
        return ["ok"]

    # 造一个声明 24000Hz 的最小 WAV 头
    wav = bytearray(64)
    wav[0:4] = b"RIFF"
    wav[8:12] = b"WAVE"
    struct.pack_into("<I", wav, 24, 24000)

    provider = DashScopeASR(api_key="sk-fake-key", sample_rate=16000)
    monkeypatch.setattr(provider, "_recognize_sync", fake_recognize_sync)
    asyncio.run(provider.atranscribe(bytes(wav), filename="a.wav"))

    assert seen["sample_rate"] == 24000, "用了配置值而不是文件里的真实采样率"


def test_raw_pcm_falls_back_to_configured_sample_rate(monkeypatch):
    """裸 PCM 没有文件头, 只能退回配置值 —— 这是唯一必须靠声明的场景。"""
    import asyncio

    from app.services.speech.dashscope_asr import DashScopeASR

    seen: dict[str, Any] = {}

    def fake_recognize_sync(
        path: str, audio_format: str = "wav", sample_rate: int | None = None
    ) -> list[str]:
        seen["sample_rate"] = sample_rate
        return ["ok"]

    provider = DashScopeASR(api_key="sk-fake-key", sample_rate=16000)
    monkeypatch.setattr(provider, "_recognize_sync", fake_recognize_sync)
    asyncio.run(provider.atranscribe(b"\x00" * 3200, filename="a.pcm"))
    assert seen["sample_rate"] == 16000


# --------------------------------------------------------------------------- #
# 结果解析: get_sentence() 的返回类型是 union, 这是最容易写错的地方
# --------------------------------------------------------------------------- #
class _FakeResult:
    def __init__(self, payload: Any, status: int = 200, message: str = "") -> None:
        self._payload = payload
        self.status_code = status
        self.message = message

    def get_sentence(self) -> Any:
        return self._payload


def test_extract_texts_from_list_payload():
    """有完整分句时 get_sentence() 返回 **list**。

    这是**正常路径**, 也恰恰是最初写错的那条: 只判断 isinstance(dict) 的话,
    识别成功时反而返回空串 —— 而"空"看起来像"用户没说话", 排查方向全跑偏。
    """
    from app.services.speech.dashscope_asr import _extract_texts

    payload = [{"text": "第一句。"}, {"text": "第二句。"}]
    assert _extract_texts(_FakeResult(payload)) == ["第一句。", "第二句。"]


def test_extract_texts_from_dict_payload():
    """只有不完整结果时返回 **dict**(直接用服务端响应的 output)。"""
    from app.services.speech.dashscope_asr import _extract_texts

    assert _extract_texts(_FakeResult({"text": "半句话"})) == ["半句话"]


def test_extract_texts_skips_empty_and_malformed_items():
    from app.services.speech.dashscope_asr import _extract_texts

    payload = [{"text": "有效"}, {"text": ""}, {"other": 1}, "不是字典", None]
    assert _extract_texts(_FakeResult(payload)) == ["有效"]


def test_extract_texts_handles_none_and_missing_fields():
    from app.services.speech.dashscope_asr import _extract_texts

    assert _extract_texts(_FakeResult(None)) == []
    assert _extract_texts(_FakeResult({})) == []
    assert _extract_texts(_FakeResult([])) == []
