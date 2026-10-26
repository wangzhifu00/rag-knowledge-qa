"""入库脚本：把一份 PDF 走完 RAG 入库线。

用法（项目根目录）：
    python scripts/ingest.py                          # 默认入库 data/langchain_intro.pdf
    python scripts/ingest.py path/to/other.pdf        # 指定文件
    python scripts/ingest.py --fresh                  # 清空向量集合重建（谨慎）

流程（2026-10-26 起原文先进 PG）：
  解析 → 清洗 → 切分
    → documents 表登记拿 doc_id
    → chunks 表整篇替换拿 chunk_ids   ← 原文的 source of truth
    → 向量化
    → Qdrant 只写 (chunk_id, 向量)
  顺序反过来了：以前是"先写 Qdrant，再在 PG 登记一行"，原文跟着向量跑；
  现在原文先落库，向量是可以随时从原文重算出来的派生数据。
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

# 让脚本能 import 项目里的 app/ 和 rag/（python 直接跑脚本时 sys.path 只有 scripts/）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.db.chunks import clear_cache, list_chunk_ids, replace_chunks
from app.db.engine import AsyncSessionLocal
from app.db.models import Document
from rag.chunker import chunk_pages
from rag.cleaner import clean_pages
from rag.embedder import get_embedder
from rag.parser import parse_pdf
from rag.retriever import (delete_points, ensure_collection, get_client,
                           upsert_chunks)


async def upsert_document_record(filename: str, file_path: str,
                                 file_size: int) -> int:
    """往 documents 表登记并返回 id（第③步建的表第一次真用上）。

    同名文件重复入库：更新记录而不是再插一行（表上有 unique 约束）。
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Document).where(Document.filename == filename)
        )
        doc = result.scalar_one_or_none()
        if doc:
            doc.file_size = file_size
            doc.file_path = file_path
            doc.status = "indexed"
            doc.error_message = None
        else:
            doc = Document(
                filename=filename,
                file_path=file_path,
                file_size=file_size,
                mime_type="application/pdf",
                status="indexed",
            )
            session.add(doc)
        await session.commit()
        await session.refresh(doc)
        return doc.id


async def main(pdf_path: str, fresh: bool = False) -> None:
    path = Path(pdf_path)
    if not path.exists():
        print(f"[错误] 文件不存在: {path}")
        sys.exit(1)

    filename = path.name
    t0 = time.time()

    # 1. 解析 + 清洗（B 阶段新增：去页脚/导航/URL/碎片行）
    pages = parse_pdf(path)
    pages = clean_pages(pages)
    print(f"[1/5] 解析+清洗完成：{len(pages)} 页有文本")

    # 2. 切分（300/30 是 golden 集评估的胜出配置，
    #    详见 scripts/eval_retrieval.py）
    chunks = chunk_pages(pages, chunk_size=300, overlap=30)
    print(f"[2/5] 切分完成：{len(chunks)} 块（chunk_size=300, overlap=30）")

    # 3. 原文落 PG（先删旧块再插新块，返回新块的 id）
    doc_id = await upsert_document_record(
        filename, str(path), path.stat().st_size)
    old_ids = await list_chunk_ids(doc_id)
    chunk_ids = await replace_chunks(doc_id, chunks)
    print(f"[3/5] chunks 表已写入 {len(chunk_ids)} 行"
          f"（doc_id={doc_id}，清掉旧块 {len(old_ids)} 个）")

    # 4. 向量化（首次运行会下载模型，请耐心）
    embedder = get_embedder()
    print(f"[4/5] 向量化中...（模型 {embedder.model_name}，维度 {embedder.dim}）")
    vectors = embedder.embed_documents([c.text for c in chunks])

    # 5. 写 Qdrant：只写 id + 向量，原文留在 PG
    client = get_client()
    # fresh 用于"存量结构变了"的一次性场景（例如这次原文从 payload 挪到 PG：
    # 老点的 id 是 chunk_index，新点的 id 是 chunks.id，两套编号会撞上）
    ensure_collection(client, dim=embedder.dim, force_recreate=fresh)
    delete_points(client, old_ids)          # 旧块在 PG 里已经不存在，向量要一起清
    count = upsert_chunks(client, chunk_ids, vectors)
    clear_cache()                            # 让 BM25 索引下次提问时重建

    dt = time.time() - t0
    print(f"[5/5] Qdrant 已写入 {count} 个点（payload 只有 chunk_id）")
    print(f"完成：耗时 {dt:.1f}s，平均 {dt / max(count, 1) * 1000:.0f}ms/块")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PDF 入库（原文进 PG，向量进 Qdrant）")
    ap.add_argument("path", nargs="?", default="data/langchain_intro.pdf",
                    help="要入库的 PDF 路径")
    ap.add_argument("--fresh", action="store_true",
                    help="清空向量集合重建（换 embedding 模型/改存量结构后用）")
    args = ap.parse_args()
    asyncio.run(main(args.path, fresh=args.fresh))
