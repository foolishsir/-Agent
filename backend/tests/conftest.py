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

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """整个测试会话共享一个应用实例(模型加载昂贵, 不能每个用例新建)."""
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP_ROOT
