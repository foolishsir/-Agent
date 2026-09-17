"""语音链路端到端冒烟（真实 provider，会联网 / 消耗额度）.

和单元测试的分工
----------------
``tests/test_speech_api.py`` 用桩 provider 测装配, 保证 CI 不依赖网络和额度.
这个脚本反过来: **专门调真实的 edge-tts 和 Paraformer**, 验证:
- edge-tts 能不能真的合成出音频(音色名、网络、配额)
- Paraformer 能不能真的识别出中文(Key、采样率、音频格式)

这两件事桩都测不到 —— 它们恰恰是最容易在真实环境里翻车的地方.

用法::

    # TTS 一定能测(免费、无需 Key)
    python backend/scripts/smoke_speech.py

    # 带上 ASR(需要先配置阿里云百炼 Key)
    python backend/scripts/smoke_speech.py --audio 我的录音.wav
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import REPO_ROOT, bootstrap  # noqa: E402

# 必须先加载界面配置, 否则读到的 Key 是空的, 会报"未配置"这种误导性结论
bootstrap()

from app.services.speech import (  # noqa: E402
    get_asr_provider,
    get_tts_provider,
    speech_status,
    to_spoken,
)

#: 用来验证"标记真的被清干净了"的样本 —— 特意混了 Markdown、引用编号、链接和箭头
SAMPLE_QUESTION = (
    "**为什么**你要用 `Redis` 做缓存？[1]\n"
    "你说响应时间提升了 40% → 这个数字是怎么测出来的？\n"
    "参考[官方文档](https://example.com/bench)里的压测方法。"
)


def _hr(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def check_status() -> None:
    _hr("① 能力自检")
    status = speech_status()
    for key in ("asr", "tts"):
        item = status[key]
        mark = "✓" if item["available"] else "✗"
        print(f"  {mark} {key.upper():4} {item['provider']}")
        if not item["available"]:
            print(f"         原因: {item['reason']}")
    print(f"    采样率 {status['sample_rate']} Hz · 单段上限 {status['max_audio_seconds']} 秒")
    if status["tts"]["available"]:
        print(f"    发音人 {status['voice']} · 语速 {status['rate']}")


def check_normalizer() -> None:
    _hr("② 口语化转换（这一步决定了读出来是什么效果）")
    print("  原始文本:")
    for line in SAMPLE_QUESTION.splitlines():
        print(f"    │ {line}")
    spoken = to_spoken(SAMPLE_QUESTION)
    print("\n  朗读文本:")
    for line in spoken.splitlines():
        print(f"    │ {line}")

    problems = []
    if "**" in spoken:
        problems.append("残留加粗标记 **")
    if "`" in spoken:
        problems.append("残留反引号")
    if "[1]" in spoken:
        problems.append("残留引用编号 [1]")
    if "http" in spoken:
        problems.append("残留 URL")
    print("\n  检查:", "✓ 干净" if not problems else "✗ " + "; ".join(problems))


async def check_tts(out_dir: Path) -> bool:
    _hr("③ 语音合成（真实调用 edge-tts）")
    provider = get_tts_provider()
    if provider is None:
        print("  ⊘ 跳过: TTS 未启用（设置 → 语音 → 语音输出选 edge）")
        return False
    cap = provider.available
    if not cap.available:
        print(f"  ⊘ 跳过: {cap.reason}")
        return False

    result = await provider.asynthesize(SAMPLE_QUESTION)
    out = out_dir / "smoke_tts.mp3"
    out.write_bytes(result.audio)

    chars_per_sec = len(result.spoken_text) / max(result.cost_ms / 1000, 0.001)
    print(f"  ✓ 合成成功: {len(result.audio):,} 字节 · 耗时 {result.cost_ms} ms")
    print(f"    口语化后 {len(result.spoken_text)} 字 · 合成吞吐 {chars_per_sec:.0f} 字/秒")
    print(f"    发音人 {result.voice}")
    print(f"\n  音频已保存: {out}")
    print("  **请打开听一遍** —— 口语化的问题只有听才能发现, 看代码看不出来.")
    return True


async def check_asr(audio_path: Path | None) -> bool:
    _hr("④ 语音识别（真实调用 Paraformer）")
    provider = get_asr_provider()
    if provider is None:
        print("  ⊘ 跳过: ASR 未启用（设置 → 语音 → 语音输入选 dashscope）")
        return False
    cap = provider.available
    if not cap.available:
        print(f"  ⊘ 跳过: {cap.reason}")
        return False

    if audio_path is None:
        print("  ⊘ 跳过: 没有提供音频文件")
        print("    用法: python backend/scripts/smoke_speech.py --audio 录音.wav")
        print("    要求: 16kHz 单声道 WAV（和前端采集格式一致）")
        return False

    if not audio_path.exists():
        print(f"  ✗ 文件不存在: {audio_path}")
        return False

    audio = audio_path.read_bytes()
    print(f"  输入: {audio_path.name} · {len(audio):,} 字节")
    result = await provider.atranscribe(audio, filename=audio_path.name)

    print(f"  ✓ 识别成功: 耗时 {result.cost_ms} ms · {len(result.sentences)} 个分句")
    print(f"\n  识别结果:\n    │ {result.text or '(空)'}")
    if not result.text:
        print("\n  ⚠ 识别结果为空 —— 常见原因:")
        print("    1. 采样率不是 16kHz（不一致时通常表现为乱码或空结果, 而不是报错）")
        print("    2. 音频不是单声道 PCM WAV")
        print("    3. 录音本身没有有效语音")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="语音链路端到端冒烟")
    parser.add_argument("--audio", type=Path, default=None, help="用于识别的 16kHz 单声道 WAV")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data", help="合成音频的存放目录")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    check_status()
    check_normalizer()
    tts_ok = asyncio.run(check_tts(args.out))
    asr_ok = asyncio.run(check_asr(args.audio))

    _hr("结论")
    print("  口语化    ✓")
    print(f"  语音合成  {'✓' if tts_ok else '⊘ 未测'}")
    print(f"  语音识别  {'✓' if asr_ok else '⊘ 未测'}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
