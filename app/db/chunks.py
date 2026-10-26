"""chunks 表的读写口：原文的 source of truth 在 PostgreSQL。

分工：
  PG     —— 原文、页码、所属文档（可查、可校验、可重建）
  Qdrant —— 向量 + point id（id 就是 chunks.id）

为什么这里有两套引擎（同一张表，一个异步一个同步）：
  写入走 async —— ingest 脚本和上传接口本来就在异步上下文里，直接用 AsyncSession。
  读取走 sync  —— 检索链路（BM25 全量扫、向量结果回填原文、rerank）是纯 CPU
                  同步逻辑。为了读一次原文把整条链路染成 async 传到底，收益为零，
                  真正的耗时（embedding/rerank）本来就要丢线程池。
  psycopg v3 两种模式都支持，共用一个 DSN，这里只是各开一个连接池。

为什么读要缓存：
  BM25 建索引需要全量原文。几百块时每次查询也还行，但提问是热路径，
  一次全表扫虽然只有几毫秒，乘以 QPS 就没必要了。缓存失效判据用
  (行数, 最大id)——重新入库是"删旧+插新"，新行 id 一定更大，所以能测出来。
"""

from dataclasses import dataclass
from functools import lru_cache
import threading

from sqlalchemy import create_engine, delete, func, insert, select
from sqlalchemy.engine import Engine

from app.config import get_settings
from app.db.engine import AsyncSessionLocal
from app.db.models import Chunk, Document
from rag.chunker import Chunk as ChunkData


@dataclass
class ChunkRow:
    """从 PG 读出来的一块（带文件名，省得调用方再查一次 documents）。"""
    id: int
    document_id: int
    chunk_index: int
    text: str
    page_numbers: list[int]
    filename: str


@lru_cache
def get_sync_engine() -> Engine:
    """同步引擎（检索层用）。

    DSN 必须显式写成 postgresql+psycopg://——SQLAlchemy 见到裸的
    postgresql:// 会去找 psycopg2，而项目装的是 psycopg v3，会直接报
    ModuleNotFoundError: psycopg2。
    """
    dsn = get_settings().postgres_dsn.replace(
        "postgresql://", "postgresql+psycopg://")
    return create_engine(dsn, pool_pre_ping=True, pool_size=4, max_overflow=4)


async def replace_chunks(document_id: int, chunks: list[ChunkData]) -> list[int]:
    """整篇替换某文档的 chunk，返回新块的 id（按 chunk_index 顺序）。

    为什么是"删完再插"而不是 upsert：重新入库时块数可能变少
    （改了切分参数、换了清洗规则），upsert 会留下一批旧尾巴。
    """
    async with AsyncSessionLocal() as session:
        await session.execute(
            delete(Chunk).where(Chunk.document_id == document_id)
        )
        if not chunks:
            await session.commit()
            return []

        rows = [
            {
                "document_id": document_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "page_numbers": c.page_numbers or [],
                "char_count": len(c.text),
            }
            for c in chunks
        ]
        # RETURNING 一次拿回所有自增 id：不用插完再查一遍（也不用担心并发）
        result = await session.execute(
            insert(Chunk).returning(Chunk.id, Chunk.chunk_index), rows
        )
        pairs = result.all()
        await session.commit()

    # RETURNING 不保证顺序，按 chunk_index 排回来才能和 vectors 一一对应
    return [chunk_id for _, chunk_id in sorted((ci, cid) for cid, ci in pairs)]


async def list_chunk_ids(document_id: int) -> list[int]:
    """取某文档现有的 chunk id（重新入库前记下来，用来清 Qdrant 里的旧向量）。"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Chunk.id).where(Chunk.document_id == document_id)
        )
        return [row[0] for row in result]


async def count_chunks(session) -> int:
    """某个 session 视角下的块数（上传接口回显用）。"""
    return (await session.execute(select(func.count()).select_from(Chunk))).scalar_one()


def _fetch_rows(ids: list[int] | None = None) -> list[ChunkRow]:
    """按 id 取块；ids=None 表示全量。内部函数，调用方走下面的缓存版。"""
    stmt = (
        select(Chunk.id, Chunk.document_id, Chunk.chunk_index,
               Chunk.text, Chunk.page_numbers, Document.filename)
        .join(Document, Chunk.document_id == Document.id)
    )
    if ids is not None:
        stmt = stmt.where(Chunk.id.in_(ids))

    with get_sync_engine().connect() as conn:
        return [
            ChunkRow(id=r.id, document_id=r.document_id,
                     chunk_index=r.chunk_index, text=r.text,
                     page_numbers=list(r.page_numbers or []), filename=r.filename)
            for r in conn.execute(stmt)
        ]


# 全量缓存：{"stamp": (行数, 最大id), "rows": [ChunkRow...]}
_corpus_cache: dict = {}
_cache_lock = threading.Lock()


def load_corpus() -> tuple[list[ChunkRow], int]:
    """全量 chunk（BM25 建索引要），返回 (所有块, 总块数)。

    缓存命中就不查库；语料一变（重新入库/删文档）自动重建。
    """
    with get_sync_engine().connect() as conn:
        stamp = conn.execute(
            select(func.count(Chunk.id), func.coalesce(func.max(Chunk.id), 0))
        ).one()
    stamp = (stamp[0], stamp[1])

    with _cache_lock:
        if _corpus_cache.get("stamp") == stamp:
            return _corpus_cache["rows"], stamp[0]

    rows = _fetch_rows()
    with _cache_lock:
        _corpus_cache.clear()
        _corpus_cache.update({"stamp": stamp, "rows": rows})
    return rows, len(rows)


def fetch_by_ids(ids: list[int]) -> dict[int, ChunkRow]:
    """按 id 批量取（向量检索回填原文用，不走缓存：命中集合每次都不一样）。"""
    if not ids:
        return {}
    return {row.id: row for row in _fetch_rows(ids)}


def clear_cache() -> None:
    """清缓存（重建索引/删文档后主动调一下，避免读到旧语料）。"""
    with _cache_lock:
        _corpus_cache.clear()
