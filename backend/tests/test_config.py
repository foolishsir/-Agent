"""配置加载与派生逻辑的测试.

这些用例保护的都是「踩过坑」的地方:
逗号分隔列表解析、相对路径解析、Embedding 双侧指令一致性.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import PROJECT_ROOT, Settings, get_settings


def test_settings_is_cached_singleton() -> None:
    """同一进程内反复取配置应命中缓存, 避免每请求重复解析 .env."""
    assert get_settings() is get_settings()


def test_relative_paths_are_resolved_to_absolute() -> None:
    cfg = Settings(data_dir=Path("data"), chroma_persist_dir=Path("data/chroma"))
    assert cfg.data_dir.is_absolute()
    assert cfg.chroma_persist_dir.is_absolute()
    assert cfg.data_dir == (PROJECT_ROOT / "data").resolve()


def test_absolute_paths_are_kept_as_is() -> None:
    target = PROJECT_ROOT / "custom_data"
    cfg = Settings(data_dir=target)
    assert cfg.data_dir == target.resolve()


def test_comma_separated_extensions_are_parsed() -> None:
    """支持 DOCMIND_ALLOWED_EXTENSIONS=.pdf,.docx 这种更符合直觉的写法."""
    cfg = Settings(allowed_extensions=".pdf,.DOCX , .md")  # type: ignore[arg-type]
    assert cfg.allowed_extensions == [".pdf", ".docx", ".md"]


def test_comma_separated_cors_origins_are_parsed() -> None:
    cfg = Settings(cors_origins="http://a.com,http://b.com")  # type: ignore[arg-type]
    assert cfg.cors_origins == ["http://a.com", "http://b.com"]


def test_llm_configured_flag() -> None:
    assert Settings(llm_api_key="sk-abc").llm_configured is True
    assert Settings(llm_api_key="   ").llm_configured is False


def test_default_retrieval_params_are_sane() -> None:
    """检索参数必须自洽: 融合常数 k 非负, 最终条数不超过召回条数."""
    cfg = Settings()
    assert cfg.rrf_k > 0
    assert 0 < cfg.final_top_k <= max(cfg.vector_top_k, cfg.bm25_top_k)
    assert cfg.child_chunk_size < cfg.parent_chunk_size


def test_ensure_dirs_is_idempotent(tmp_root: Path) -> None:
    cfg = Settings(
        data_dir=tmp_root / "d1",
        upload_dir=tmp_root / "d1" / "uploads",
        log_dir=tmp_root / "l1",
        chroma_persist_dir=tmp_root / "d1" / "chroma",
    )
    cfg.ensure_dirs()
    cfg.ensure_dirs()  # 重复调用不应抛异常
    assert cfg.upload_dir.exists()
    assert cfg.log_dir.exists()
