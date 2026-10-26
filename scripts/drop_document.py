"""按文件名删除一份文档：PG 里的文档记录 + chunk 原文 + Qdrant 里的向量。

为什么需要这个脚本：
  1. 接口层只有上传和列表，没有删除——传错文件、想重传新版时很尴尬
  2. 评估必须跑在**受控语料**上。实测过一个坑：往库里塞了一份简历后，
     bge-small 给「DeepAgents是什么」返回的 top1 竟是简历段落（余弦 0.539），
     于是 RAGAS 的 context_precision/recall 全变 0——指标掉的是语料，不是模型

原文进 PG 之后，删除的顺序变成"先按 PG 查出 chunk id，再按 id 删 Qdrant 点，
最后删 documents 行（PG 外键 ON DELETE CASCADE 自动带走 chunks）"。
反过来做的话，Qdrant 点会变成找不到原文的孤儿向量。

回滚很简单：重新上传同一份 PDF 即可（本地 embedding 不花钱）。

用法（项目根目录）：
    python scripts/drop_document.py 王植富.pdf          # 只删除这一份
    python scripts/drop_document.py 王植富.pdf --dry-run  # 只看会删多少块
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, select

from app.db.chunks import clear_cache
from app.db.engine import AsyncSessionLocal
from app.db.models import Chunk, Document
from rag.retriever import delete_points, get_client


async def find_document(filename: str) -> tuple[int, list[int]] | None:
    """按文件名找到 doc_id 和它的全部 chunk id（找不到返回 None）。"""
    async with AsyncSessionLocal() as db:
        doc = (await db.execute(
            select(Document).where(Document.filename == filename)
        )).scalar_one_or_none()
        if doc is None:
            return None
        ids = (await db.execute(
            select(Chunk.id).where(Chunk.document_id == doc.id)
        )).scalars().all()
        return doc.id, list(ids)


async def drop_row(filename: str) -> int:
    """删 documents 行；chunks 由外键 ON DELETE CASCADE 自动清掉。"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            delete(Document).where(Document.filename == filename)
        )
        await db.commit()
        return result.rowcount or 0


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="按文件名删除文档（PG 记录 + chunk 原文 + Qdrant 向量）")
    parser.add_argument("filename", help="要删除的文件名，如 王植富.pdf")
    parser.add_argument("--dry-run", action="store_true", help="只统计不删除")
    args = parser.parse_args()

    found = await find_document(args.filename)
    if found is None:
        print(f"[WARN] PostgreSQL 里没有「{args.filename}」，检查文件名是否与库里完全一致")
        return

    doc_id, chunk_ids = found
    print(f"文档「{args.filename}」：doc_id={doc_id}，"
          f"chunks 表 {len(chunk_ids)} 块，对应 Qdrant {len(chunk_ids)} 个点")

    if args.dry_run:
        print("[DRY-RUN] 未做任何修改")
        return

    delete_points(get_client(), chunk_ids)
    print(f"[OK] 已从 Qdrant 删除 {len(chunk_ids)} 个点")

    rows = await drop_row(args.filename)
    clear_cache()  # BM25 缓存里还留着这份文档的块，得清掉
    print(f"[OK] 已删除 documents {rows} 行（chunks 随外键级联删除）")

    # BM25 是进程内缓存，按语料版本（行数, 最大id）自动重建，
    # 这里清完缓存后下一次检索就会重建，不需要重启服务。
    print("提示：BM25 索引会在下次检索时自动重建；正在跑的服务无需重启")


if __name__ == "__main__":
    asyncio.run(main())
