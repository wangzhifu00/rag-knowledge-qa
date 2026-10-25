"""
一次性探测应用依赖的 3 个外部组件，把结果汇总成一个 JSON 返回
判断"服务现在能不能正常工作"
"""

import asyncio
import logging
import httpx
import psycopg
import redis.asyncio as aioredis
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.config import Settings, get_settings

router = APIRouter(tags=["health"])

logger = logging.getLogger(__name__)

class ComponentStatus(BaseModel):
    name: str # 组件名
    ok: bool   # 是否健康
    detail: str = ""  # 失败原因/附加信息

class HealthResponse(BaseModel):
    status: str   # "ok" 或 "degraded"
    components: list[ComponentStatus]   # 每个组件一条

#尝试连接/请求 → 成功返回 ok=True → 任何异常都捕获 → 返回 ok=False 并带上错误信息。

# psycopg3 的异步连接
async def check_postgres(settings: Settings) -> ComponentStatus:
    try:
        async with await psycopg.AsyncConnection.connect(
            settings.postgres_dsn, connect_timeout=3  #connect_timeout=3 防止数据库挂了时这个请求卡死。
        ) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                await cur.fetchone()
        return ComponentStatus(name="postgres", ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("postgres 健康检查失败: %s", exc)
        return ComponentStatus(name="postgres", ok=False, detail=str(exc)[:200])


async def check_redis(settings: Settings) -> ComponentStatus:
    client = aioredis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password or None,
        socket_connect_timeout=3,
    )
    try:
        pong = await client.ping()
        return ComponentStatus(name="redis", ok=bool(pong))
    except Exception as exc:  # noqa: BLE001
        logger.warning("redis 健康检查失败: %s", exc)
        return ComponentStatus(name="redis", ok=False, detail=str(exc)[:200])
    finally:
        await client.aclose()   #不管成功失败都关连接，避免每次健康检查泄漏一个连接池


async def check_qdrant(settings: Settings) -> ComponentStatus:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{settings.qdrant_url}/healthz")
        return ComponentStatus(name="qdrant", ok=resp.status_code == 200,
                               detail=f"HTTP {resp.status_code}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("qdrant 健康检查失败: %s", exc)
        return ComponentStatus(name="qdrant", ok=False, detail=str(exc)[:200])

#任何一个组件挂了，不应该让整个健康检查接口 500，而是要报告哪个挂了。

@router.get("/health", response_model=HealthResponse, summary="依赖组件健康检查")
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """三个依赖并发探测，任何一个挂了 status 就是 degraded。"""
    components = await asyncio.gather(
        check_postgres(settings),
        check_redis(settings),
        check_qdrant(settings),
    )

    all_ok = all(c.ok for c in components)   #全绿才是 "ok"，任何一个失败 → "degraded"
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        components=list(components),
    )
