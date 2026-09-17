"""用浏览器同款 WAV 打一次真实的 /speech/transcribe, 看服务端怎么理解它.

这一步验证的是**浏览器录音路径**上的三件事:
  1. format 判断成了 wav
  2. 采样率从文件头读出了 16000(而不是配置里的值)
  3. Paraformer 能接受这个容器(不报 decode 错误)

音频本身是合成信号没有语音, 所以**识别结果为空是正确的** ——
这一条测的是"能不能进得去", 不是"识别得准不准".
"""

from __future__ import annotations

import json
import pathlib
import urllib.error
import urllib.request
import uuid

BASE = "http://127.0.0.1:8000/api/v1"
WAV = pathlib.Path(__file__).resolve().parents[2] / "data" / "browser_shape.wav"


def main() -> int:
    audio = WAV.read_bytes()
    print(f"输入: {WAV.name} · {len(audio):,} 字节")

    # 验证文件头确实是 16kHz
    import struct

    rate = struct.unpack_from("<I", audio, 24)[0]
    rate = int(rate)
    print(f"文件头采样率: {rate} Hz")
    assert rate == 16000, "文件头不是 16kHz"

    b = uuid.uuid4().hex
    body = b"".join(
        [
            f'--{b}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="answer.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode(),
            audio,
            f"\r\n--{b}--\r\n".encode(),
        ]
    )
    req = urllib.request.Request(
        BASE + "/speech/transcribe",
        data=body,
        headers={"X-User-Id": "demo-user", "Content-Type": f"multipart/form-data; boundary={b}"},
        method="POST",
    )

    try:
        payload = json.loads(urllib.request.urlopen(req, timeout=120).read())["data"]
    except urllib.error.HTTPError as exc:
        print(f"\n✗ HTTP {exc.code}")
        print(json.loads(exc.read()).get("message", ""))
        return 1

    print("\n服务端看到的是:")
    print(f"  audio_format = {payload['audio_format']}")
    print(f"  sample_rate  = {payload['sample_rate']}")
    print(f"  provider     = {payload['provider']}")
    print(f"  cost_ms      = {payload['cost_ms']}")
    print(f"  empty        = {payload['empty']}")
    print(f"  text         = {payload['text']!r}")

    ok = True
    if payload["audio_format"] != "wav":
        print("  ✗ format 判断错了")
        ok = False
    if payload["sample_rate"] != 16000:
        print(f"  ✗ 采样率读成了 {payload['sample_rate']}, 应该是 16000")
        ok = False
    if not payload["empty"]:
        print("  ! 合成信号居然识别出了内容, 有点意外(不算失败)")

    print()
    print("结论: 浏览器录音路径" + ("贯通" if ok else "仍有问题"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
