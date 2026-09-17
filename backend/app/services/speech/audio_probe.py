"""从音频文件头读出真实采样率（零依赖）.

为什么必须做这件事
------------------
Paraformer 会校验**声明的采样率**和**文件里的真实采样率**是否一致, 不一致直接报:

    语音识别失败(44): Failed to decode audio: sample rate 16000 not equals with real 24000

这条坑在真实链路上立刻撞到了: edge-tts 合成出来的是 **24kHz** mp3,
而 ``settings.speech_asr_sample_rate`` 是 16000(给浏览器录音用的),
把配置值直接传给 SDK 就炸了.

为什么不用现成的库
------------------
读采样率只需要看几十个字节的头:
- **WAV**: 头是固定布局, 采样率在偏移 24 处, 4 字节小端
- **MP3**: 第一个帧头 4 字节里, 第 10~11 位是采样率索引

为这点事引入 ``soundfile`` / ``pydub`` / ``ffmpeg`` 不划算 ——
尤其 ffmpeg 是个 70MB 的外部二进制, 和项目"零安装"的主张冲突
(这也是前端不用 MediaRecorder 的同一个理由).

读不出来的格式(m4a/opus/aac)返回 None, 由调用方退回配置值 ——
**宁可退回配置值也不要猜**: 猜错会得到乱码而不是报错, 更难查.
"""

from __future__ import annotations

import struct

#: MPEG 版本 → 采样率表. 索引来自帧头的第 10~11 位.
_MP3_RATES: dict[int, tuple[int, int, int]] = {
    3: (44100, 48000, 32000),  # MPEG 1
    2: (22050, 24000, 16000),  # MPEG 2
    0: (11025, 12000, 8000),  # MPEG 2.5
}

#: 帧头里采样率索引为 3 表示"保留", 是非法的
_MP3_INVALID_INDEX = 3


def _probe_wav(data: bytes) -> int | None:
    """从 WAV 头读采样率.

    标准布局::

        0  "RIFF"  4 文件长度  8 "WAVE"
        12 "fmt "  16 子块长度(16)  20 格式(1=PCM)  22 声道  24 **采样率**
        28 字节率  32 块对齐  34 位深

    不做完整的 chunk 遍历 —— 我们**自己生成**的 WAV 一定是标准布局.
    对第三方 WAV 若偏移不对, 读到离谱的值会被下面的合理性检查挡掉.
    """
    if len(data) < 28 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    rate = struct.unpack_from("<I", data, 24)[0]
    return rate if _is_sane(rate) else None


def _probe_mp3(data: bytes) -> int | None:
    """从 MP3 第一个帧头读采样率.

    帧头 4 字节(大端位序)::

        AAAAAAAA AAABBCCD EEEEFFGH IIJJKLMM
        A: 同步字(全 1)   B: MPEG 版本   C: 层   F: **采样率索引**

    前 512 字节里找第一个合法的帧同步 —— 文件开头可能有 ID3 标签,
    直接读前 4 字节会读到 "ID3\\x03" 而不是帧头. 这是最常见的实现错误.
    """
    limit = min(len(data), 512)
    # 上界是 limit - 3 而不是 limit - 4: 帧头 4 字节从 i 开始,
    # 最后一个合法起点是 limit-4, 而 range 的上界是开区间.
    # 写成 range(limit - 4) 会同时犯两个错 —— 4 字节输入时循环体一次都不执行,
    # 且永远扫描不到紧贴末尾的同步字.
    for i in range(max(0, limit - 3)):
        if data[i] != 0xFF or (data[i + 1] & 0xE0) != 0xE0:
            continue  # 不是同步字
        version = (data[i + 1] >> 3) & 0x03
        rate_index = (data[i + 2] >> 2) & 0x03
        # 版本 1 是保留值; 采样率索引 3 也是保留值 —— 出现说明这是误命中的同步字
        if version == 1 or rate_index == _MP3_INVALID_INDEX:
            continue
        rates = _MP3_RATES.get(version)
        if rates:
            return rates[rate_index]
    return None


def _is_sane(rate: int) -> bool:
    """合理性检查.

    读错偏移时会读出天文数字或 0. 语音采样率不会超过 192kHz ——
    与其把一个离谱的值传给 ASR(它只会报"不一致"),
    不如识别成"读不出来"然后退回配置值.
    """
    return 8000 <= rate <= 192000


def probe_sample_rate(data: bytes, filename: str = "") -> int | None:
    """尽力从音频头部读出真实采样率.

    Args:
        data: 音频字节
        filename: 用于判断格式(取后缀); 传空则按内容特征猜测

    Returns:
        采样率, 或 None 表示这个格式读不出来
    """
    if not data:
        return None

    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix == "wav" or data[0:4] == b"RIFF":
        return _probe_wav(data)
    if suffix in ("mp3", "mp2") or data[0:3] == b"ID3":
        return _probe_mp3(data)
    # 其余格式(m4a / opus / aac / amr / pcm)不做解析:
    # 它们的头部结构复杂得多, 而我们的主路径用不到
    return None


__all__ = ["probe_sample_rate"]
