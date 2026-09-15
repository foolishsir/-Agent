"""FastAPI 依赖注入.

把鉴权、数据库会话这类"每个接口都要用"的东西收敛到这里,
接口函数只声明自己需要什么, 不关心怎么构造.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db

#: 数据库会话依赖
SessionDep = Annotated[AsyncSession, Depends(get_db)]


async def get_current_user_id(
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> str:
    """获取当前用户 id.

    当前是**占位实现**: 从请求头读 ``X-User-Id``, 缺省为 ``default-user``.
    之所以这么做而不是硬编码一个常量, 是为了让"多用户数据隔离"这条链路
    从现在起就是真实可测的 —— 换个请求头就能验证隔离是否生效.

    P4 阶段会替换成 JWT 解析: 从 token 里取 ``sub`` 作为 user_id,
    并且拒绝无 token 的请求. 届时接口层代码**不需要改动**,
    因为依赖的签名不变 —— 这正是依赖注入的价值.
    """
    return (x_user_id or "").strip() or "default-user"


CurrentUser = Annotated[str, Depends(get_current_user_id)]
