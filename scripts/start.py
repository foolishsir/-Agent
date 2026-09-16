"""一键启动 —— 检查环境后启动服务, 并自动打开浏览器.

用法::

    python scripts/start.py              # 默认 127.0.0.1:8000
    python scripts/start.py --port 8001
    python scripts/start.py --no-browser

启动前会做四项检查, 把常见问题挡在启动之前 ——
否则用户看到的是一个打不开的页面, 完全不知道哪里错了.
"""

from __future__ import annotations

import argparse
import importlib.util
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from contextlib import suppress
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

REQUIRED_PACKAGES = {
    "fastapi": "Web 框架",
    "uvicorn": "ASGI 服务器",
    "fitz": "PDF 解析",
    "chromadb": "向量数据库",
    "sentence_transformers": "本地向量模型",
}


def setup_console() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            with suppress(Exception):
                stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def title(text: str) -> None:
    print()
    print("=" * 62)
    print(f"  {text}")
    print("=" * 62)
    print()


def ok(text: str) -> None:
    print(f"  ✓ {text}")


def warn(text: str) -> None:
    print(f"  ! {text}")


def fail(text: str) -> None:
    print(f"  ✗ {text}")


# --------------------------------------------------------------------------- #
# 启动前检查
# --------------------------------------------------------------------------- #
def check_dependencies() -> bool:
    missing = [name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        fail("依赖缺失: " + ", ".join(missing))
        print()
        print("  请先运行安装脚本:")
        print("    Windows :  双击 install.bat")
        print("    其他系统:  python scripts/setup.py")
        return False

    ok("依赖就绪")
    return True


def check_env() -> bool:
    """检查配置文件, 返回是否已配置 Key."""
    if not ENV_FILE.exists():
        warn("未找到 .env, 正在从模板创建")
        if ENV_EXAMPLE.exists():
            ENV_FILE.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            fail("模板也不存在, 请检查项目文件是否完整")
            return False

    key = ""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("DOCMIND_LLM_API_KEY="):
            key = stripped.split("=", 1)[1].strip()

    if key:
        ok(f"LLM API Key 已配置 (末四位 {key[-4:]})")
        return True

    print()
    warn("尚未配置 LLM API Key")
    print("     服务可以正常启动 —— 文档上传、分块预览都能用,")
    print("     但「智能问答」会提示未配置 Key。")
    print("     启动后在网页「设置」页面填入即可, 不用重启。")
    return False


def port_in_use(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def resolve_port(host: str, port: int, *, auto_next: bool) -> int | None:
    if not port_in_use(host, port):
        return port

    warn(f"端口 {port} 已被占用")
    print("     可能是上次启动的服务还在运行。")

    if auto_next:
        for candidate in range(port + 1, port + 10):
            if not port_in_use(host, candidate):
                ok(f"自动改用端口 {candidate}")
                return candidate
        fail("连续 10 个端口都被占用")
        return None

    try:
        answer = input(f"     改用端口 {port + 1} 启动吗? [Y/n] ").strip().lower()
    except EOFError:
        answer = "n"
    if answer in ("", "y", "yes"):
        return port + 1

    print("     已取消。")
    return None


def open_browser_later(url: str, delay: float, ready_event: threading.Event) -> None:
    """延迟打开浏览器, 并等模型预热完成后再刷新一次.

    首次启动要下载并预热模型(1~3 分钟), 这段时间端口还没开始接受连接 ——
    浏览器会显示"无法连接". 所以这里等就绪信号, 到了再打开,
    用户看到的直接就是能用的页面.
    """
    deadline = time.time() + delay + 300  # 最多等 5 分钟
    while time.time() < deadline:
        if ready_event.is_set():
            break
        time.sleep(0.5)

    with suppress(Exception):
        webbrowser.open(url)


def wait_for_ready(host: str, port: int, ready_event: threading.Event) -> None:
    """轮询端口, 一旦可连接就发出就绪信号(后台线程)."""
    deadline = time.time() + 600
    while time.time() < deadline:
        if port_in_use(host, port):
            # 端口通了还要等一下, 让应用完成 lifespan 启动
            time.sleep(1.0)
            ready_event.set()
            return
        time.sleep(0.5)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> int:
    setup_console()

    parser = argparse.ArgumentParser(description="启动 DocMind 服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--reload", action="store_true", help="开启热重载(开发用)")
    parser.add_argument("--auto-port", action="store_true", help="端口被占用时自动换一个, 不询问")
    args = parser.parse_args()

    title("DocMind 启动服务")

    if not check_dependencies():
        return 1

    has_key = check_env()

    port = resolve_port(args.host, args.port, auto_next=args.auto_port)
    if port is None:
        return 1
    ok(f"端口 {port} 可用")

    url = f"http://{args.host}:{port}"
    print()
    print("=" * 62)
    print("  正在启动 ...")
    print("=" * 62)
    print()
    print("  首次启动会下载并预热本地模型, 大约 1~3 分钟。")
    print("  **在出现「DocMind 已就绪」之前浏览器打不开, 这是正常的。**")
    print()
    print(f"  访问地址: {url}")
    print("  接口文档: " + url + "/docs")
    print("  示例文档: samples/ 目录 (可直接拖进上传区)")
    print()
    print("  停止服务: 按 Ctrl+C, 或直接关闭本窗口")
    print("=" * 62)
    print()

    ready_event = threading.Event()
    if not args.no_browser:
        threading.Thread(
            target=wait_for_ready, args=(args.host, port, ready_event), daemon=True
        ).start()
        threading.Thread(
            target=open_browser_later, args=(url, 3.0, ready_event), daemon=True
        ).start()
        print("  (服务就绪后会自动打开浏览器)")
        print()
    elif has_key:
        pass

    # 直接用 uvicorn 启动, 并把标准输出继承给当前终端 ——
    # 用户需要看到启动日志和后续的请求日志
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        args.host,
        "--port",
        str(port),
    ]
    if args.reload:
        cmd.append("--reload")

    try:
        return subprocess.call(cmd, cwd=str(BACKEND))
    except KeyboardInterrupt:
        print("\n  已停止。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
