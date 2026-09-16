"""pytest 全局 fixture.

关键点: 在导入 app 之前设置环境变量, 保证测试跑在独立的
临时数据目录上, 不会污染开发环境的 data/ 与 Chroma 库.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 必须在 `from app...` 之前设置: 配置对象在模块导入时就会被实例化.
# ---------------------------------------------------------------------------
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="docmind-test-"))
os.environ.setdefault("DOCMIND_APP_ENV", "test")
os.environ.setdefault("DOCMIND_DEBUG", "true")
os.environ.setdefault("DOCMIND_DATA_DIR", str(_TMP_ROOT / "data"))
os.environ.setdefault("DOCMIND_UPLOAD_DIR", str(_TMP_ROOT / "data" / "uploads"))
os.environ.setdefault("DOCMIND_LOG_DIR", str(_TMP_ROOT / "logs"))
os.environ.setdefault("DOCMIND_CHROMA_PERSIST_DIR", str(_TMP_ROOT / "data" / "chroma"))
os.environ.setdefault("DOCMIND_CHROMA_MODE", "embedded")
os.environ.setdefault("DOCMIND_TASK_MODE", "inline")
# 测试不预热模型: 预热的意义是把加载开销前移, 测试里只会白白拖慢每一次运行
os.environ.setdefault("DOCMIND_WARMUP_ON_STARTUP", "false")

# ---------------------------------------------------------------------------
# 数据库也必须指向临时目录!
#
# 这是一个真实踩到的严重问题: 最初只改了 data_dir / chroma 目录, 却漏了
# DATABASE_URL. 由于 config.py 会把 sqlite 的相对路径解析到"项目根目录",
# 测试实际上一直在往**开发库 data/docmind.db 里写数据** ——
# 表现是"某个用例的计数断言偶尔多 1", 极难联想到是测试污染了开发数据.
#
# 这条必须在导入 app 之前设置. 注意用 as_posix(): sqlite URL 必须是正斜杠形式.
# ---------------------------------------------------------------------------
(_TMP_ROOT / "data").mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "DOCMIND_DATABASE_URL",
    f"sqlite+aiosqlite:///{(_TMP_ROOT / 'data' / 'docmind.db').as_posix()}",
)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402


class FakeEmbeddingProvider:
    """确定性的假 Embedding, 让测试不依赖真实模型.

    为什么需要它:
    1. 真实模型加载要几秒, 每个测试都加载会让测试套件慢到没人愿意跑
    2. 真实模型的输出不可预测, 断言无法稳定 —— 测试必须**确定性**
    3. 单元测试的边界应该是"入库流程是否正确", 不是"BGE 模型准不准"

    真实模型的正确性由 ``scripts/parse_pdf.py`` 和标记为 slow 的用例覆盖.
    """

    dim = 8

    @property
    def name(self) -> str:
        return "fake:deterministic"

    def _vector(self, text: str) -> list[float]:
        """用文本的哈希生成稳定向量, 保证同样输入永远得到同样输出."""
        import hashlib
        import math

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i] / 255.0 - 0.5 for i in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture(scope="session", autouse=True)
def fake_embedding() -> FakeEmbeddingProvider:
    """全局替换 Embedding 提供方与精排器.

    **必须逐个替换模块自己命名空间里的引用**. 这些模块在顶部做了
    ``from app.services.embedding import get_embedding_provider``,
    也就是说它们在**导入时**就把函数对象绑到了自己的模块命名空间里.
    只替换原始模块的话, 这些模块仍然指向旧函数 ——
    测试会静默地加载真实模型(实测单个用例耗时 19 秒),
    而"测试跑得慢"这件事很容易被当成正常现象而放过.

    精排器同样返回 None: 未启用时检索链路会退化为按融合分数截断,
    结果依然确定且有意义, 但省掉了几秒的模型加载.
    真实精排模型由手工端到端验证覆盖, 而不是放在单元测试里.
    """
    from app.services import ingest
    from app.services.retrieval import pipeline as retrieval_pipeline

    provider = FakeEmbeddingProvider()

    patches: list[tuple[object, str, object]] = [
        (ingest, "get_embedding_provider", lambda: provider),
        (retrieval_pipeline, "get_embedding_provider", lambda: provider),
        (retrieval_pipeline, "get_reranker", lambda: None),
    ]

    originals = [(target, name, getattr(target, name)) for target, name, _ in patches]
    for target, name, replacement in patches:
        setattr(target, name, replacement)

    yield provider

    for target, name, original in originals:
        setattr(target, name, original)


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """整个测试会话共享一个应用实例(避免反复建表与初始化向量库)."""
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP_ROOT


@pytest.fixture(scope="session")
def sample_pdf(tmp_root: Path) -> Path:
    """生成一份**结构可预测**的测试 PDF.

    刻意构造了三种真实文档里常见的干扰, 用来验证解析器的处理能力:

    1. **每页重复的页眉与页码页脚** —— 应该被跨页重复检测剔除
    2. **字号分层** —— 18pt 大标题 / 12pt 章节标题 / 10pt 正文
    3. **多页** —— 跨页重复检测需要至少 3 页才有统计意义

    用英文正文是为了让 PyMuPDF 的内置字体稳定输出可提取的文本层;
    中文相关的清洗与分块逻辑用纯字符串在单元测试里覆盖, 不依赖 PDF 渲染.
    """
    import pymupdf as fitz

    path = tmp_root / "sample.pdf"
    doc = fitz.open()

    sections = [
        (
            "Equipment Maintenance Guide",
            "Section One: Cutter Blade",
            "cutter blade service life is 20000 cycles or 3 months.",
        ),
        (
            "Equipment Maintenance Guide",
            "Section Two: Solder Paste",
            "solder paste must be stored between 2 and 10 degrees celsius.",
        ),
        (
            "Equipment Maintenance Guide",
            "Section Three: Cleaning",
            "the cleaning station runs a weekly calibration routine.",
        ),
    ]

    for page_no, (title, heading, body) in enumerate(sections, start=1):
        page = doc.new_page(width=595, height=842)
        # 页眉: 三页完全一致 → 应被识别为 running header 并剔除
        page.insert_text((72, 40), title, fontsize=8)
        # 大标题: 只在第 1 页出现.
        # 如果每页都放, 它就会被跨页重复检测当成页眉删掉 —— 这正是该机制的预期行为,
        # 但会让"标题识别"这个用例测不到东西.
        if page_no == 1:
            page.insert_text((72, 100), "DocMind Test Document", fontsize=18)
        # 章节标题(每页不同, 不会被误判为页眉)
        page.insert_text((72, 140), heading, fontsize=12)
        # 正文(多行, 用于验证断行合并)
        page.insert_text((72, 170), f"This document states that the {body}", fontsize=10)
        page.insert_text((72, 185), "It also records the maintenance interval for", fontsize=10)
        page.insert_text((72, 200), "each station in the production line.", fontsize=10)
        # 页脚: 页码每页不同, 但归一化后一致 → 也应被剔除
        page.insert_text((290, 810), f"Page {page_no} of 3", fontsize=8)

    doc.save(str(path))
    doc.close()
    return path
