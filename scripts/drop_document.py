"""按文件名删除一份文档：Qdrant 里的所有块 + PostgreSQL 的文档记录。

为什么需要这个脚本：
  1. 接口层只有上传和列表，没有删除——传错文件、想重传新版时很尴尬
  2. 评估必须跑在**受控语料**上。实测过一个坑：往库里塞了一份简历后，
     bge-small 给「DeepAgents是什么」返回的 top1 竟是简历段落（余弦 0.539），
     于是 RAGAS 的 context_precision/recall 全变 0——指标掉的是语料，不是模型
  
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

from qdrant_client import models
from sqlalchemy import delete, select

from app.db.engine import AsyncSessionLocal
from app.db.models import Document
from rag.retriever import COLLECTION_NAME, get_client


def count_points(filename: str) -> int:
    client = get_client()
    return client.count(
        collection_name=COLLECTION_NAME,
        count_filter=models.Filter(must=[
            models.FieldCondition(key="filename", match=models.MatchValue(value=filename))
        ]),
        exact=True,
    ).count


def drop_points(filename: str) -> None:
    """按 payload.filename 删点（Qdrant 的过滤删除，不动其他文档）。"""
    get_client().delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[
                models.FieldCondition(key="filename", match=models.MatchValue(value=filename))
            ])
        ),
    )


async def drop_row(filename: str) -> int:
    async with AsyncSessionLocal() as db:
        result = await db.execute(delete(Document).where(Document.filename == filename))
        await db.commit()
        return result.rowcount or 0


async def main() -> None:
    parser = argparse.ArgumentParser(description="按文件名删除文档（Qdrant 块 + PG 记录）")
    parser.add_argument("filename", help="要删除的文件名，如 王植富.pdf")
    parser.add_argument("--dry-run", action="store_true", help="只统计不删除")
    args = parser.parse_args()

    n = count_points(args.filename)
    print(f"集合 {COLLECTION_NAME} 中含有 {n} 个属于「{args.filename}」的块")

    if n == 0:
        print("[WARN] Qdrant 里没有匹配的块，检查文件名是否与库里完全一致")

    if args.dry_run:
        print("[DRY-RUN] 未做任何修改")
        return

    drop_points(args.filename)
    print(f"[OK] 已从 Qdrant 删除 {n} 个块")

    rows = await drop_row(args.filename)
    print(f"[OK] 已从 PostgreSQL Document 表删除 {rows} 行")

    # 提醒：BM25 是进程内缓存，按集合点数变化自动重建，
    # 这里删完点后，下一次检索会自动重建索引，不需要重启服务。
    print("提示：BM25 索引会在下次检索时按新块数自动重建；正在跑的服务无需重启")


if __name__ == "__main__":
    asyncio.run(main())
