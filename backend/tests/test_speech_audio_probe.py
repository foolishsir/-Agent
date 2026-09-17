"""音频头解析的单元测试.

这个模块存在的唯一理由: Paraformer 会校验**声明的采样率**和**文件里的真实采样率**,
不一致直接报错. 而我们的两个音频来源采样率天然不同:

    浏览器录音（WAV）    16 kHz
    edge-tts 合成（MP3） 24 kHz

把配置值无条件传过去就会炸 —— 真实撞到过。所以必须从文件头读真实值。

测试用的样本都是**手工拼的最小头部**, 不带真实音频数据 ——
解析器只看前几十个字节, 不需要完整文件, 这样测试快且不依赖任何外部资源.
"""

from __future__ import annotations

import struct

import pytest

from app.services.speech.audio_probe import probe_sample_rate


def make_wav_bytes(sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    """拼一个最小的合法 WAV 头(44 字节)."""
    buf = bytearray(44)
    buf[0:4] = b"RIFF"
    buf[8:12] = b"WAVE"
    buf[12:16] = b"fmt "
    struct.pack_into("<I", buf, 16, 16)  # fmt 块长度
    struct.pack_into("<H", buf, 20, 1)  # PCM
    struct.pack_into("<H", buf, 22, channels)
    struct.pack_into("<I", buf, 24, sample_rate)
    byte_rate = sample_rate * channels * bits // 8
    struct.pack_into("<I", buf, 28, byte_rate)
    struct.pack_into("<H", buf, 32, channels * bits // 8)
    struct.pack_into("<H", buf, 34, bits)
    buf[36:40] = b"data"
    return bytes(buf)


def make_mp3_frame(version_bits: int, rate_index: int) -> bytes:
    """拼一个 MP3 帧头(4 字节).

    version_bits: 3=MPEG1, 2=MPEG2, 0=MPEG2.5
    rate_index:   0..2 合法, 3 是保留值
    """
    b1 = 0xFF
    b2 = 0xE0 | (version_bits << 3) | (0b01 << 1) | 0b1  # 同步 + 版本 + Layer III + 无 CRC
    b3 = (rate_index << 2) | 0b00
    b4 = 0x00
    return bytes([b1, b2, b3, b4])


# --------------------------------------------------------------------------- #
# WAV
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000])
def test_reads_wav_sample_rate(rate):
    assert probe_sample_rate(make_wav_bytes(rate), "a.wav") == rate


def test_wav_detected_by_content_even_without_filename():
    """没有文件名时按内容特征判断 —— RIFF 魔数足够可靠。"""
    assert probe_sample_rate(make_wav_bytes(16000), "") == 16000


def test_rejects_wav_like_garbage():
    """不是 RIFF 就别当 WAV 解析, 否则会读到随机字节当采样率。"""
    assert probe_sample_rate(b"XXXX" + b"\x00" * 40, "a.wav") is None


def test_rejects_insane_wav_sample_rate():
    """偏移读错会得到离谱的值.

    与其把一个天文数字传给 ASR(它只会回一句"不一致"), 不如判定"读不出来"
    然后退回配置值 —— 报错信息会更贴近真实原因.
    """
    data = bytearray(make_wav_bytes(16000))
    struct.pack_into("<I", data, 24, 999_999_999)
    assert probe_sample_rate(bytes(data), "a.wav") is None


# --------------------------------------------------------------------------- #
# MP3
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("version_bits", "rate_index", "expected"),
    [
        (3, 0, 44100),  # MPEG1
        (3, 1, 48000),
        (3, 2, 32000),
        (2, 0, 22050),  # MPEG2
        (2, 1, 24000),  # ← edge-tts 用的就是这个
        (2, 2, 16000),
        (0, 0, 11025),  # MPEG2.5
    ],
)
def test_reads_mp3_sample_rate(version_bits, rate_index, expected):
    assert probe_sample_rate(make_mp3_frame(version_bits, rate_index), "a.mp3") == expected


def test_mp3_skips_id3_tag():
    """文件开头的 ID3 标签必须跳过。

    edge-tts 输出的 mp3 **带 ID3v2 头**, 直接读前 4 字节会读到 "ID3\\x03"
    而不是帧同步字 —— 这是这类解析器最常见的实现错误。
    """
    id3 = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 20
    data = id3 + make_mp3_frame(2, 1)
    assert probe_sample_rate(data, "a.mp3") == 24000


def test_mp3_rejects_reserved_values():
    """版本=1 和采样率索引=3 都是保留值, 出现说明是误命中的同步字。

    不排除的话会返回一个不存在的采样率, 传下去就是 SDK 报"不一致"。
    """
    assert probe_sample_rate(make_mp3_frame(1, 1), "a.mp3") is None
    assert probe_sample_rate(make_mp3_frame(2, 3), "a.mp3") is None


def test_mp3_returns_none_when_no_frame_found():
    assert probe_sample_rate(b"\x00" * 600, "a.mp3") is None


# --------------------------------------------------------------------------- #
# 边界
# --------------------------------------------------------------------------- #
def test_empty_input():
    assert probe_sample_rate(b"", "a.wav") is None


def test_unsupported_format_returns_none():
    """不认识的格式返回 None, 由调用方退回配置值。

    刻意**不去猜** —— 猜错会得到乱码而不是报错, 比明确的回退更难查。
    """
    assert probe_sample_rate(b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 40, "a.m4a") is None
    assert probe_sample_rate(b"OggS" + b"\x00" * 40, "a.opus") is None


def test_short_input_does_not_crash():
    assert probe_sample_rate(b"RIFF", "a.wav") is None
    assert probe_sample_rate(b"\xff", "a.mp3") is None
    assert probe_sample_rate(b"\xff\xe0", "a.mp3") is None
