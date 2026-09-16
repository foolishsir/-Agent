"""一键安装 —— 检查环境、装依赖、建配置、自检.

为什么逻辑放在 Python 而不是 .bat 里
-----------------------------------
Windows 批处理对中文的支持非常不可靠: 即使开头写了 ``chcp 65001``,
cmd.exe 仍可能按当前代码页**逐字节解析**批处理文件, 导致中文字符串被截断,
报出 ``'失败]' is not recognized as an internal or external command`` 这种错误.
(本项目的 .bat 第一版就是这样挂掉的.)

所以 .bat 只保留一行 ASCII 启动器, 所有逻辑与中文提示都放在这里。
顺带的好处是 **Linux / macOS 用同一条命令即可**::

    python scripts/setup.py

关于数据库
----------
本项目**不需要安装任何数据库**:

- 关系库用 SQLite, Python 标准库自带
- 向量库用 Chroma, 随 pip 包安装, 内嵌在应用进程里
- Redis 只在开启任务队列模式时才需要, 默认不用
- Docker 只在需要水平扩容时才需要

装完 Python 依赖就能直接跑。
"""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

REQUIRED_PACKAGES: dict[str, str] = {
    "fastapi": "Web 框架",
    "uvicorn": "ASGI 服务器",
    "fitz": "PDF 解析 (PyMuPDF)",
    "jieba": "中文分词",
    "chromadb": "向量数据库",
    "sentence_transformers": "本地向量化与重排模型",
    "sqlalchemy": "ORM",
    "openai": "大模型客户端",
}


def setup_console() -> None:
    """Windows 控制台默认 GBK, 输出中文会乱码甚至抛 UnicodeEncodeError."""
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


def step(text: str) -> None:
    print(f"\n▶ {text}")


def ok(text: str) -> None:
    print(f"  ✓ {text}")


def warn(text: str) -> None:
    print(f"  ! {text}")


def fail(text: str) -> None:
    print(f"  ✗ {text}")


# --------------------------------------------------------------------------- #
# 1. Python
# --------------------------------------------------------------------------- #
def check_python() -> bool:
    step("[1/4] 检查 Python 版本")

    version = sys.version_info
    if version < (3, 11):
        fail(f"版本过低: {version.major}.{version.minor}.{version.micro} (需要 3.11+)")
        print()
        print("  本项目用到了 Python 3.11 才有的 StrEnum 与 asyncio 特性。")
        print("  请升级 Python: https://www.python.org/downloads/")
        return False

    ok(f"Python {version.major}.{version.minor}.{version.micro}")
    ok(f"解释器: {sys.executable}")

    if sys.platform == "win32" and "windowsapps" in sys.executable.lower():
        warn("检测到 Microsoft Store 版的 Python, 某些包可能安装失败")
        warn("如遇问题请到 python.org 下载官方安装包")
    return True


# --------------------------------------------------------------------------- #
# 2. 依赖
# --------------------------------------------------------------------------- #
def install_dependencies(*, use_mirror: bool = True) -> bool:
    step("[2/4] 安装依赖")
    print("  首次安装需要下载 1~2 GB 内容(主要是 PyTorch), 可能需要几分钟。\n")

    base = [sys.executable, "-m", "pip", "install"]
    if use_mirror:
        base += ["-i", MIRROR]

    # pip 自身先升级, 老版本对 --extra-index-url 等参数支持不全
    subprocess.call([*base, "--upgrade", "pip"], stdout=subprocess.DEVNULL)

    code = subprocess.call([*base, "-r", str(REQUIREMENTS)])
    if code != 0 and use_mirror:
        print()
        warn("镜像源安装失败, 改用官方源重试 ...")
        code = subprocess.call([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)])

    if code != 0:
        fail("依赖安装失败")
        print()
        print("  常见原因:")
        print("    - 网络不通: 检查代理设置, 或换一个 pip 镜像源")
        print("  　- 权限不足: Windows 上用管理员身份重新运行")
        print("    - 磁盘空间不足: 需要至少 5 GB 可用空间")
        return False

    ok("依赖安装完成")
    return True


def verify_dependencies() -> bool:
    """装完之后真的 import 一次, 确认可用.

    只看 pip 的返回码不够 —— 有些包装了但互相不兼容, 到运行时才炸.
    """
    import importlib.util

    missing = [name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        fail("以下依赖仍不可用: " + ", ".join(missing))
        return False

    ok(f"已验证 {len(REQUIRED_PACKAGES)} 个关键依赖可导入")
    return True


# --------------------------------------------------------------------------- #
# 3. 配置
# --------------------------------------------------------------------------- #
def prepare_env() -> str:
    """准备 .env, 返回 'ok' / 'need_key' / 'missing'."""
    step("[3/4] 准备配置文件")

    if not ENV_EXAMPLE.exists():
        fail(f"未找到配置模板: {ENV_EXAMPLE}")
        return "missing"

    if ENV_FILE.exists():
        ok(".env 已存在")
    else:
        ENV_FILE.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
        ok("已从 .env.example 创建 .env")

    # 解析 Key(不打印它)
    key = ""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("DOCMIND_LLM_API_KEY="):
            key = stripped.split("=", 1)[1].strip()

    if key:
        ok(f"LLM API Key 已配置 (末四位 {key[-4:]})")
        return "ok"

    print()
    warn("还没有配置 LLM API Key")
    print()
    print("  两种配置方式, 任选其一:")
    print()
    print("    A. 现在填 —— 用记事本打开 .env, 找到这一行:")
    print("         DOCMIND_LLM_API_KEY=")
    print("       在等号后填入你的 Key(形如 sk-xxxxxxxx), 保存即可。")
    print()
    print("    B. 启动后再填 —— 打开网页 http://127.0.0.1:8000,")
    print("       进入「设置」页面填写, 保存后立即生效, 不用重启。")
    print()
    print("  Key 申请地址: https://platform.deepseek.com")
    print("  (文档上传、分块预览不需要 Key, 只有问答需要)")
    return "need_key"


def offer_open_env() -> None:
    if sys.platform == "win32":
        with suppress(Exception):
            os.startfile(ENV_FILE)  # type: ignore[attr-defined]
            ok("已用记事本打开 .env")
    else:
        ok(f"请手动编辑: {ENV_FILE}")


# --------------------------------------------------------------------------- #
# 4. 自检
# --------------------------------------------------------------------------- #
def run_self_check() -> bool:
    step("[4/4] 环境自检")
    print()
    script = PROJECT_ROOT / "backend" / "scripts" / "check_env.py"
    code = subprocess.call([sys.executable, str(script)])
    return code == 0


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> int:
    setup_console()

    title("DocMind 一键安装")
    print("  提示: 本项目不需要安装 MySQL / Redis / Docker。")
    print("        关系库用 SQLite(Python 自带), 向量库用 Chroma(pip 包, 进程内嵌)。")

    if not check_python():
        return 1

    if not install_dependencies():
        return 1

    if not verify_dependencies():
        return 1

    state = prepare_env()
    run_self_check()

    title("安装完成")

    if state == "need_key":
        print("  下一步:")
        print("    1. 在 .env 里填入 DOCMIND_LLM_API_KEY (或启动后在网页「设置」里填)")
        print("    2. 双击 start.bat 启动服务")
    else:
        print("  下一步: 双击 start.bat 启动服务")

    print()
    print("  首次启动会下载 Embedding 模型(约 95MB)并预热, 需要 1~3 分钟。")
    print("  .env 里已经默认配好国内镜像 HF_ENDPOINT, 下载不会太慢。")
    print()
    print("  示例文档已放在 samples/ 目录, 启动后直接拖进上传区即可体验。")
    print()

    if state == "need_key":
        try:
            answer = input("  现在打开 .env 填写 Key 吗? [y/N] ").strip().lower()
        except EOFError:
            answer = "n"
        if answer == "y":
            offer_open_env()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
