"""日志体系.

设计要点
--------
1. **TraceId 贯穿全链路**: 通过 ``contextvars`` 保存当前请求的 trace_id,
   由 ``TraceIdFilter`` 自动注入到每条日志中. 这样一次上传+解析+向量化+问答
   的全过程日志可以用同一个 id 串起来, 排查线上问题效率完全不同.
2. **控制台彩色 + 文件轮转**: 控制台给人看, 文件按大小轮转给人事后查.
3. 提供 ``bind_trace_id`` 供后台任务(不在请求上下文中)手动绑定链路 id.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import uuid
from contextlib import suppress
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from app.core.config import settings

# 当前请求/任务的链路 id. ContextVar 在 asyncio 下天然按协程隔离,
# 不会出现多请求并发时 trace_id 串号的问题.
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(trace_id)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
}
_RESET = "\033[0m"


def new_trace_id() -> str:
    """生成一个新的短链路 id."""
    return uuid.uuid4().hex[:12]


def get_trace_id() -> str:
    return _trace_id_var.get()


def set_trace_id(trace_id: str | None = None) -> str:
    """设置当前上下文链路 id, 返回最终生效的值."""
    value = trace_id or new_trace_id()
    _trace_id_var.set(value)
    return value


class TraceIdFilter(logging.Filter):
    """把 contextvar 中的 trace_id 注入到日志记录里."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id_var.get()
        return True


class ColorFormatter(logging.Formatter):
    """控制台彩色格式化器(仅对 levelname 着色)."""

    def __init__(self, fmt: str, datefmt: str, use_color: bool = True) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if not self.use_color:
            return super().format(record)
        color = _COLORS.get(record.levelname, "")
        original = record.levelname
        record.levelname = f"{color}{original}{_RESET}"
        try:
            return super().format(record)
        finally:
            # 恢复原始值, 否则同一个 record 被多个 handler 处理时会重复着色
            record.levelname = original


def setup_logging(level: str | int | None = None, log_dir: Path | None = None) -> None:
    """初始化根 logger. 幂等, 重复调用只会重置 handler."""
    # Windows 控制台默认 GBK 编码, 中文日志会乱码甚至抛 UnicodeEncodeError
    # 直接把请求打断. 统一转成 UTF-8 输出.
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            with suppress(Exception):
                stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    resolved_level = level or (logging.DEBUG if settings.debug else logging.INFO)
    if isinstance(resolved_level, str):
        resolved_level = logging.getLevelName(resolved_level.upper())

    directory = log_dir or settings.log_dir
    directory.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(resolved_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    trace_filter = TraceIdFilter()

    # --- 控制台 ---
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(resolved_level)
    console.addFilter(trace_filter)
    console.setFormatter(ColorFormatter(_LOG_FORMAT, _DATE_FORMAT, use_color=sys.stdout.isatty()))
    root.addHandler(console)

    # --- 文件(轮转, 单文件 10MB, 保留 5 份) ---
    file_handler = logging.handlers.RotatingFileHandler(
        filename=directory / "docmind.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(resolved_level)
    file_handler.addFilter(trace_filter)
    file_handler.setFormatter(ColorFormatter(_LOG_FORMAT, _DATE_FORMAT, use_color=False))
    root.addHandler(file_handler)

    # 降噪: 第三方库的日志太吵.
    #
    # 尤其是 DEBUG 级别下的 aiosqlite / sqlalchemy.engine ——
    # 它们会把**每一条 SQL 和每一个游标操作**都打出来. 实测一次文档入库
    # 能产生上千行日志, 把自己的业务日志彻底淹没.
    # 排查 ORM 问题时可以临时把这两个名字从列表里去掉, 但不要长期开着.
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "chromadb",
        "sentence_transformers",
        "modelscope",
        "aiosqlite",
        "sqlalchemy.engine",
        "sqlalchemy.pool",
        "asyncio",
        "multipart",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> logging.Logger:
    """获取 logger. 建议传 ``__name__``."""
    return logging.getLogger(name or "docmind")


def log_kv(logger: logging.Logger, event: str, **kwargs: Any) -> None:
    """结构化打点. 便于后续接入日志采集系统做检索耗时/召回率分析.

    用法::

        log_kv(logger, "retrieval.done", stage="vector", top_k=20, cost_ms=31.2)
    """
    if not kwargs:
        logger.info(event)
        return
    pairs = " ".join(f"{key}={value}" for key, value in kwargs.items())
    logger.info("%s | %s", event, pairs)
