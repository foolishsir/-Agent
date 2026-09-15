"""数据库引擎与会话管理.

设计要点
--------
1. **引擎是进程级单例**. 每次请求新建引擎会反复建立连接池, 是典型的性能事故.
2. **SQLite 必须手动打开外键约束**. SQLite 默认 FOREIGN KEY 是关闭的,
   不打开的话 ``ON DELETE CASCADE`` 形同虚设, 删文档不会级联删分块.
   这是一个非常隐蔽的坑 —— 代码看起来完全正确, 但产生了孤儿数据.
3. **会话不自动过期对象**(``expire_on_commit=False``). 否则 commit 之后再访问
   对象属性会触发一次额外的 SELECT(甚至因会话已关闭而报错),
   在 async 场景下这个问题更明显.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logging import get_logger
from app.models.base import Base

logger = get_logger("docmind.db")

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """获取(或惰性创建)全局异步引擎."""
    global _engine
    if _engine is None:
        is_sqlite = settings.database_url.startswith("sqlite")

        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            future=True,
            # SQLite 的写操作是串行的, 连接池开大没有意义, 反而更容易触发
            # "database is locked". NullPool/小池 + 较长的 busy timeout 才是正解.
            pool_pre_ping=not is_sqlite,
            connect_args={"timeout": 30} if is_sqlite else {},
        )

        if is_sqlite:
            _enable_sqlite_foreign_keys(_engine)

        logger.info("数据库引擎已创建 | url=%s", _mask_dsn(settings.database_url))

    return _engine


def _enable_sqlite_foreign_keys(engine: AsyncEngine) -> None:
    """为每个 SQLite 连接打开外键约束.

    必须在 ``connect`` 事件里逐连接执行: PRAGMA 是**连接级**设置,
    建库时执行一次是不够的, 之后新开的连接依然是关闭状态.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            # WAL 模式: 读写不互相阻塞, 明显缓解 SQLite 的并发写问题
            cursor.execute("PRAGMA journal_mode=WAL")
        finally:
            cursor.close()


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """获取会话工厂."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def init_db() -> None:
    """建表.

    生产环境应该用 Alembic 做版本化迁移, 这里用 ``create_all`` 是为了
    让项目能"clone 下来直接跑" —— 降低首次运行门槛对这个项目很重要.
    如果表结构开始频繁变更, 就应该引入 Alembic 了.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("数据表已就绪 | tables=%s", ", ".join(sorted(Base.metadata.tables)))


async def dispose_engine() -> None:
    """释放连接池(应用关闭时调用)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("数据库连接池已释放")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖: 每个请求一个会话.

    异常时不吞掉异常而是回滚后重新抛出 —— 否则上层异常处理器拿到的是
    一个"成功"的假象, 会被转成 200 响应.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def check_connection() -> tuple[bool, str]:
    """连通性探测, 供就绪探针使用."""
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True, _mask_dsn(settings.database_url)
    except Exception as exc:  # noqa: BLE001 - 探针需要兜住一切
        return False, f"连接失败: {exc}"


def _mask_dsn(url: str) -> str:
    """隐藏 DSN 中的密码后再打日志."""
    if "@" not in url or "://" not in url:
        return url
    scheme, _, rest = url.partition("://")
    if "@" not in rest:
        return url
    credentials, _, host = rest.rpartition("@")
    user = credentials.split(":")[0]
    return f"{scheme}://{user}:***@{host}"
