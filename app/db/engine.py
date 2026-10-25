"""数据库连接层：engine + session 工厂。

- engine：全局唯一的连接池管理者（懒连接，不在这里真正连库）
- AsyncSessionLocal：会话工厂，每个请求开一个 session，用完自动还回池子
- get_db：FastAPI 依赖注入用的生成器，路由函数写 db: AsyncSession = Depends(get_db)
"""

from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from app.config import get_settings

settings = get_settings()

# pool_pre_ping=True：每次从池里拿连接前先探一下还活着没，防止 PG 重启后拿到死连接
engine = create_async_engine(
    settings.postgres_dsn_async,
    pool_pre_ping=True,
    echo=False,  # True 会把每条 SQL 打到控制台，调试时可以开
)

# expire_on_commit=False：commit 之后不把对象标记为过期，
# 避免"commit 后再访问属性又偷偷发一次查询"的坑
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：一个请求一个 session，请求结束自动关闭。"""
    async with AsyncSessionLocal() as session:
        yield session
