"""Alembic 迁移环境。
关键点：
1. 从 app.config 读数据库连接（单一事实来源，不在 alembic.ini 里重复写）
2. 用异步引擎跑迁移（和运行时保持一致，都走 asyncpg）
3. target_metadata = Base.metadata，这样 autogenerate 才能对比
   "模型定义 vs 数据库现状"，自动生成差异脚本
"""

import asyncio
from logging.config import fileConfig
from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.config import get_settings
from app.db.models import Base

# Alembic 的配置对象
config = context.config

# 让 alembic.ini 里的日志配置生效
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 把项目的全部表结构交给 Alembic（autogenerate 的对比基准）
target_metadata = Base.metadata

# 从应用配置注入连接串（同步写 ini 的方式在这里不适用，直接 set）
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.postgres_dsn_async)

def run_migrations_offline() -> None:
    """离线模式：只生成 SQL 不连库（alembic --sql 时用）。"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection: Connection) -> None:
    """在真实连接上执行迁移（被下面的 run_sync 回调）。"""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """异步模式：建一个临时引擎，跑完即弃。"""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
