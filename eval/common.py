"""评测公共工具: 工作区隔离、评测集读写、文档入库.

**工作区隔离很重要**: 评测要反复用不同分块参数重新入库同一份文档,
如果跑在开发库里, 会把开发数据覆盖掉、也会被已有数据干扰指标.
所以评测使用**完全独立的数据目录 / SQLite / Chroma 集合**,
跑完可以直接删掉 ``eval/.workspace``.

环境变量必须在导入 app 之前设置 —— 配置对象在模块导入时就会实例化.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import AsyncIterator
from pathlib import Path

#: 项目根目录 (eval/ 的上一级)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
EVAL_DIR = PROJECT_ROOT / "eval"

# 两个路径都要加:
# - BACKEND: 导入 app.* 包
# - PROJECT_ROOT: 以 `eval.metrics` 的形式导入评测模块(而不是同名裸模块),
#   避免 metrics.py 与别的模块重名时导入到错误的文件
for _p in (str(BACKEND), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_WORKSPACE = EVAL_DIR / ".workspace"
GOLDEN_SET = EVAL_DIR / "golden_set.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

#: 开发环境的运行时配置(在 Web 界面「设置」里保存的).
#: 评测工作区是隔离的, 读不到它 —— 但 LLM Key 通常就配在这里,
#: 所以评测启动时把它**复制**一份过去, 而不是让用户再配一遍.
DEV_RUNTIME_CONFIG = PROJECT_ROOT / "data" / "runtime_settings.json"

#: 评测用例的 user_id. 与真实用户隔离, 便于必要时的清理
EVAL_USER = "eval-user"


def setup_workspace(
    workspace: Path = DEFAULT_WORKSPACE,
    *,
    reset: bool = False,
    inherit_runtime_config: bool = True,
) -> Path:
    """把应用指向评测专用工作区.

    **必须在导入任何 app.* 模块之前调用** —— 配置是模块级单例, 导入即定型.
    """
    if reset and workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)

    (workspace / "data").mkdir(parents=True, exist_ok=True)

    os.environ["DOCMIND_DATA_DIR"] = str(workspace / "data")
    os.environ["DOCMIND_UPLOAD_DIR"] = str(workspace / "data" / "uploads")
    os.environ["DOCMIND_LOG_DIR"] = str(workspace / "logs")
    os.environ["DOCMIND_CHROMA_PERSIST_DIR"] = str(workspace / "data" / "chroma")
    os.environ["DOCMIND_DATABASE_URL"] = (
        f"sqlite+aiosqlite:///{(workspace / 'data' / 'eval.db').as_posix()}"
    )
    # 用独立的集合名, 避免和开发环境的向量数据混在一起
    os.environ["DOCMIND_CHROMA_COLLECTION"] = "docmind_eval"
    os.environ["DOCMIND_TASK_MODE"] = "inline"
    os.environ["DOCMIND_WARMUP_ON_STARTUP"] = "false"

    # 继承界面上配好的 LLM 参数(Key / base_url / 模型名).
    # 只复制文件, 不复制其它数据 —— LLM 之外的配置由评测脚本自己控制,
    # 否则"评测跑的是哪个参数"会变得不可控.
    if inherit_runtime_config and DEV_RUNTIME_CONFIG.exists():
        shutil.copy2(DEV_RUNTIME_CONFIG, workspace / "data" / "runtime_settings.json")

    return workspace


def apply_runtime_config() -> int:
    """把运行时配置应用到 settings 单例, 并建好目录.

    必须在导入 app.* 之后调用(配置对象在导入时创建, 之后才能改它的值).

    顺带调用 ``ensure_dirs()``: 正常启动流程里这是 lifespan 干的活,
    但评测脚本是直接调服务层, 不走 lifespan —— 不建目录的话第一次上传
    就会因为 uploads 不存在而失败.
    """
    from app.core.config import settings  # noqa: PLC0415
    from app.services import config_service  # noqa: PLC0415

    settings.ensure_dirs()
    return config_service.load_runtime_overrides()


# --------------------------------------------------------------------------- #
# 评测集
# --------------------------------------------------------------------------- #
def load_golden_set(path: Path | None = None) -> list:
    """读取 JSONL 格式的评测集.

    ``path`` 省略时用默认路径. 注意要判 ``None`` 而不是用 ``or`` 兜底 ——
    显式传入不存在的路径时应该报错, 而不是悄悄回退到默认文件,
    否则"我明明指定了另一个评测集, 结果跑的还是默认的"会很难发现.
    """
    from eval.metrics import GoldenItem  # noqa: PLC0415

    path = Path(path) if path is not None else GOLDEN_SET
    if not path.exists():
        raise FileNotFoundError(
            f"评测集不存在: {path}\n先运行: python eval/generate_golden_set.py --pdf <你的文档.pdf>"
        )

    items: list[GoldenItem] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"评测集第 {lineno} 行不是合法 JSON: {exc}") from exc

        items.append(
            GoldenItem(
                id=str(data["id"]),
                question=str(data["question"]),
                reference_answer=str(data.get("reference_answer", "")),
                evidences=[str(e) for e in data.get("evidences", []) if str(e).strip()],
                category=str(data.get("category", "semantic")),
                source_pages=[int(p) for p in data.get("source_pages", [])],
                doc_ids=[str(d) for d in data.get("doc_ids", [])],
            )
        )

    if not items:
        raise ValueError(f"评测集是空的: {path}")
    return items


def save_golden_set(items: list, path: Path = GOLDEN_SET) -> None:
    from dataclasses import asdict  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for item in items:
        data = asdict(item)
        lines.append(json.dumps(data, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def describe_golden_set(items: list) -> dict[str, object]:
    """评测集概况 —— 跑之前先看一眼分布是否健康."""
    from collections import Counter  # noqa: PLC0415

    categories = Counter(i.category for i in items)
    return {
        "total": len(items),
        "categories": dict(categories),
        "answerable": sum(1 for i in items if not i.is_no_answer),
        "no_answer": sum(1 for i in items if i.is_no_answer),
        "with_evidence": sum(1 for i in items if i.evidences),
    }


# --------------------------------------------------------------------------- #
# 文档入库(复用真实服务, 而不是另写一套)
# --------------------------------------------------------------------------- #
async def ingest_pdf(pdf_path: Path, *, user_id: str = EVAL_USER) -> str:
    """把 PDF 入库到评测工作区, 返回 doc_id.

    刻意走**真实的** ``document_service`` + ``run_ingest``, 而不是评测专用的简化路径.
    评测的意义就是衡量"真实系统"的表现; 如果评测跑的是另一套代码,
    测出来的数字对线上毫无参考价值 —— 这是评测体系里最常见的自欺欺人.
    """
    from app.db.session import get_session_factory  # noqa: PLC0415
    from app.services import document_service  # noqa: PLC0415
    from app.services.ingest import submit_ingest  # noqa: PLC0415

    payload = pdf_path.read_bytes()

    async def stream() -> AsyncIterator[bytes]:
        yield payload

    session_factory = get_session_factory()
    async with session_factory() as session:
        document, _created = await document_service.create_document(
            session, filename=pdf_path.name, stream=stream(), user_id=user_id
        )
        doc_id = document.id

    result = await submit_ingest(doc_id)
    if result.status.value != "READY":
        raise RuntimeError(f"文档入库失败: {result.error}")

    return doc_id


async def reset_eval_data() -> None:
    """清空评测库的关系数据(向量库由 workspace reset 处理)."""
    from sqlalchemy import delete  # noqa: PLC0415

    from app.db.session import get_session_factory  # noqa: PLC0415
    from app.models import Chunk, Conversation, Document, Message  # noqa: PLC0415

    async with get_session_factory()() as session:
        await session.execute(delete(Message))
        await session.execute(delete(Conversation))
        await session.execute(delete(Chunk))
        await session.execute(delete(Document))
        await session.commit()


def print_banner(title: str) -> None:
    width = 78
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def ensure_utf8_console() -> None:
    """Windows 控制台默认 GBK, 打印中文会乱码甚至抛 UnicodeEncodeError."""
    if sys.platform == "win32":
        from contextlib import suppress  # noqa: PLC0415

        for stream in (sys.stdout, sys.stderr):
            with suppress(Exception):
                stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
