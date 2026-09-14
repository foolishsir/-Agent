"""环境自检脚本 —— 把「跑不起来」的问题一次性暴露出来.

用法::

    cd docmind
    python backend/scripts/check_env.py

为什么需要它: RAG 项目的失败原因 80% 不在代码, 而在环境 ——
缺依赖、模型没下载、Key 没配、镜像没设. 与其让用户跑到一半报错,
不如启动前花两秒全量体检一遍.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import socket
import sys
from contextlib import suppress
from pathlib import Path

# Windows 控制台默认用 GBK(cp936) 编码, 打印中文/emoji 会乱码甚至抛
# UnicodeEncodeError. 强制 UTF-8 输出, 保证在任何终端下都能看清体检结果.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        with suppress(Exception):
            _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

PASS = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"

_results: list[tuple[str, str, str]] = []


def record(level: str, name: str, detail: str) -> None:
    _results.append((level, name, detail))


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n {title}\n{'=' * 72}")


# --------------------------------------------------------------------------- #
# 1. 运行环境
# --------------------------------------------------------------------------- #
def check_runtime() -> None:
    section("1. 运行环境")
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 11)
    record(PASS if ok else FAIL, "Python 版本", f"{platform.python_version()} (需要 >= 3.11)")
    record(PASS, "操作系统", f"{platform.system()} {platform.release()}")
    record(PASS, "解释器路径", sys.executable)

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        detail = f"torch {torch.__version__}, CUDA {'可用' if cuda_ok else '不可用'}"
        if cuda_ok:
            detail += f", GPU={torch.cuda.get_device_name(0)}"
        record(PASS, "PyTorch", detail)
    except ImportError:
        record(WARN, "PyTorch", "未安装(本地 Embedding 需要, 云端 API 模式可忽略)")


# --------------------------------------------------------------------------- #
# 2. 依赖包
# --------------------------------------------------------------------------- #
REQUIRED = {
    "fastapi": "Web 框架",
    "uvicorn": "ASGI 服务器",
    "pydantic": "数据校验",
    "pydantic_settings": "配置管理",
    "multipart": "文件上传",
    "fitz": "PDF 解析 (PyMuPDF)",
    "jieba": "中文分词",
    "rank_bm25": "关键词检索",
    "chromadb": "向量数据库",
    "openai": "LLM 客户端",
    "sqlalchemy": "ORM",
    "aiosqlite": "SQLite 异步驱动",
}

OPTIONAL = {
    "sentence_transformers": "本地 Embedding / Rerank",
    "sse_starlette": "SSE 流式输出",
    "redis": "任务队列 (queue 模式需要)",
    "rq": "任务队列 Worker",
    "tenacity": "重试",
    "aiofiles": "异步文件写入",
}


def check_packages() -> None:
    section("2. 依赖包")
    for module, purpose in REQUIRED.items():
        found = importlib.util.find_spec(module) is not None
        record(PASS if found else FAIL, module, purpose if found else f"缺失 —— {purpose}")

    print()
    for module, purpose in OPTIONAL.items():
        found = importlib.util.find_spec(module) is not None
        record(PASS if found else WARN, module, purpose if found else f"缺失 —— {purpose}")


# --------------------------------------------------------------------------- #
# 3. 配置
# --------------------------------------------------------------------------- #
def check_config() -> None:
    section("3. 配置 (.env)")
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        record(PASS, ".env", str(env_file))
    else:
        record(FAIL, ".env", "不存在 —— 请执行 `copy .env.example .env` 并填写")
        return

    try:
        from app.core.config import settings
    except Exception as exc:  # noqa: BLE001 - 自检脚本需要兜住一切
        record(FAIL, "配置加载", f"失败: {exc}")
        return

    record(PASS, "应用环境", f"{settings.app_env} (debug={settings.debug})")

    if settings.llm_configured:
        key = settings.llm_api_key
        record(
            PASS, "LLM", f"{settings.llm_provider}/{settings.llm_model} key={key[:4]}***{key[-4:]}"
        )
    else:
        record(FAIL, "LLM", "未配置 DOCMIND_LLM_API_KEY —— 问答接口会返回 503")

    record(
        PASS,
        "Embedding",
        f"{settings.embedding_provider}/{settings.embedding_model} dim={settings.embedding_dim}",
    )
    record(
        PASS,
        "Rerank",
        settings.rerank_model if settings.rerank_enabled else "已关闭",
    )
    record(PASS, "向量库", f"chroma/{settings.chroma_mode} @ {settings.chroma_persist_dir}")
    record(PASS, "任务模式", settings.task_mode)

    # --- 目录可写 ---
    print()
    try:
        settings.ensure_dirs()
        probe = settings.data_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        record(PASS, "数据目录", f"{settings.data_dir} (可写)")
        record(PASS, "上传目录", str(settings.upload_dir))
        record(PASS, "日志目录", str(settings.log_dir))
    except Exception as exc:  # noqa: BLE001
        record(FAIL, "数据目录", f"不可写: {exc}")

    # --- HF 镜像 ---
    print()
    endpoint = os.environ.get("HF_ENDPOINT", "")
    if endpoint:
        record(PASS, "HF 镜像", endpoint)
    else:
        record(
            WARN,
            "HF 镜像",
            "未设置 HF_ENDPOINT —— 国内首次下载模型可能超时, 建议设为 https://hf-mirror.com",
        )


# --------------------------------------------------------------------------- #
# 4. 模型缓存
# --------------------------------------------------------------------------- #
def check_model_cache() -> None:
    section("4. 本地模型缓存")
    if importlib.util.find_spec("sentence_transformers") is None:
        record(WARN, "模型缓存", "sentence-transformers 未安装, 跳过检查")
        return

    try:
        from app.core.config import settings
    except Exception:  # noqa: BLE001
        return

    hf_home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    hub = hf_home / "hub"
    for model in (settings.embedding_model, settings.rerank_model):
        if not settings.rerank_enabled and model == settings.rerank_model:
            continue
        slug = "models--" + model.replace("/", "--")
        cached = (hub / slug).exists()
        record(
            PASS if cached else WARN,
            model,
            f"已缓存 @ {hub / slug}" if cached else "未缓存 —— 首次使用时会自动下载",
        )


# --------------------------------------------------------------------------- #
# 5. 外部服务
# --------------------------------------------------------------------------- #
def check_services() -> None:
    section("5. 外部服务")

    for name, host, port in (("Redis", "localhost", 6379), ("Chroma Server", "localhost", 8001)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        try:
            reachable = sock.connect_ex((host, port)) == 0
        finally:
            sock.close()
        record(
            PASS if reachable else WARN,
            name,
            f"{host}:{port} 可连接"
            if reachable
            else f"{host}:{port} 未监听 —— embedded/inline 模式下属正常",
        )


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def print_report() -> int:
    section("检查结果汇总")
    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]

    for level, name, detail in _results:
        print(f"{level} {name:<24} {detail}")

    print(
        f"\n通过 {len(_results) - len(fails) - len(warns)} 项 | "
        f"警告 {len(warns)} 项 | 失败 {len(fails)} 项"
    )

    if fails:
        print("\n必须先解决以下问题:")
        for _, name, detail in fails:
            print(f"  - {name}: {detail}")
        print("\n修复建议:")
        print(
            "  1. 安装依赖: pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"
        )
        print("  2. 复制配置: copy .env.example .env")
        print("  3. 填写 DOCMIND_LLM_API_KEY (https://platform.deepseek.com)")
        return 1

    if warns:
        print("\n以上警告不影响启动, 但部分功能可能不可用.")
    print("\n环境就绪, 可以启动服务: cd backend && python -m uvicorn app.main:app --reload")
    return 0


def main() -> int:
    print("DocMind 环境自检")
    check_runtime()
    check_packages()
    check_config()
    check_model_cache()
    check_services()
    return print_report()


if __name__ == "__main__":
    raise SystemExit(main())
