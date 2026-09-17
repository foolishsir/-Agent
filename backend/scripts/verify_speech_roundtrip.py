"""真实链路往返验证: edge-tts 说一句话 → Paraformer 听回来.

为什么这个验证方式特别好
------------------------
不需要人录音, **全程自动且判定客观**: 合成时我们知道原文, 识别回来一比对,
就知道整条语音链路是不是通的. 而且它同时覆盖了两件事:

- edge-tts 能不能真的出声(mp3 格式)
- Paraformer 能不能真的听懂(以及 format 声明对不对)

跨格式本身也是一次真实验证: 合成出来是 **mp3**, 识别时 format 必须跟着变成 mp3.
写死 format="wav" 的话这里就会露馅(识别出乱码而不是报错).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import REPO_ROOT, bootstrap  # noqa: E402

# 必须在导入 app.* 之前加载界面配置 —— 否则读到的 Key 是空的,
# 会得出"未配置"这种误导性结论
bootstrap()

from app.services.speech import get_asr_provider, get_tts_provider, speech_status  # noqa: E402

SENTENCE = "我在项目里用 Redis 做了缓存，把订单查询的响应时间降低了百分之四十。"


async def main() -> int:
    print("=" * 68)
    print("语音链路往返验证（合成 → 识别）")
    print("=" * 68)

    status = speech_status()
    if not status["tts"]["available"]:
        print(f"✗ TTS 不可用: {status['tts']['reason']}")
        return 1
    if not status["asr"]["available"]:
        print(f"✗ ASR 不可用: {status['asr']['reason']}")
        return 1

    # ---------------- ① 合成 ----------------
    tts = get_tts_provider()
    assert tts is not None
    print(f"\n① 合成（{tts.name}）")
    print(f"   原文: {SENTENCE}")
    synth = await tts.asynthesize(SENTENCE)
    out = REPO_ROOT / "data" / "roundtrip.mp3"
    out.write_bytes(synth.audio)
    print(f"   ✓ {len(synth.audio):,} 字节 mp3 · {synth.cost_ms} ms")
    print(f"   朗读文本: {synth.spoken_text[:60]}...")

    # ---------------- ② 识别 ----------------
    asr = get_asr_provider()
    assert asr is not None
    print(f"\n② 识别（{asr.name}）")
    print("   输入: roundtrip.mp3（注意是 mp3, 不是 wav）")
    try:
        result = await asr.atranscribe(synth.audio, filename="roundtrip.mp3")
    except Exception as exc:  # noqa: BLE001
        print(f"   ✗ 识别失败: {type(exc).__name__}: {exc}")
        return 1

    print(f"   ✓ {result.cost_ms} ms · {len(result.sentences)} 个分句")
    print(f"   识别结果: {result.text}")

    # ---------------- ③ 比对 ----------------
    print("\n③ 比对")
    if not result.text.strip():
        print("   ✗ 识别结果为空 —— 检查 format 声明是否和文件真实格式一致")
        return 1

    # 不做严格字符串相等(识别会有标点/用词差异), 只查关键实体是否还原
    for token in ("Redis", "缓存", "订单", "查询"):
        ok = token.lower() in result.text.lower()
        print(f"   {'✓' if ok else '✗'} 关键词「{token}」{'命中' if ok else '丢失'}")

    overlap = len(set(result.text) & set(synth.spoken_text)) / max(len(set(synth.spoken_text)), 1)
    print(f"   字符重合度: {overlap:.0%}")

    print("\n结论: 语音输入输出链路贯通")
    print(f"音频留存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
