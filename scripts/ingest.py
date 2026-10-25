"""入库脚本：把一份 PDF 走完 RAG 入库线。

用法（项目根目录）：
    python scripts/ingest.py                          # 默认入库 data/langchain_intro.pdf
    python scripts/ingest.py path/to/other.pdf        # 指定文件

流程：解析 → 切分 → 向量化 → 存 Qdrant → 在 documents 表登记
"""

import asyncio
import sys
import time
from pathlib import Path

# 让脚本能 import 项目里的 app/ 和 rag/（python 直接跑脚本时 sys.path 只有 scripts/）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from app.db.engine import AsyncSessionLocal
from app.db.models import Document
from rag.chunker import chunk_pages
from rag.cleaner import clean_pages
from rag.embedder import get_embedder
from rag.parser import parse_pdf
from rag.retriever import (
    delete_points_of_file,
    ensure_collection,
    get_client,
    upsert_chunks,
)


async def save_document_record(filename: str, file_path: str, file_size: int,
                               chunk_count: int) -> None:
    """往 documents 表登记（第③步建的表第一次真用上）。

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
            session.add(Document(
                filename=filename,
                file_path=file_path,
                file_size=file_size,
                mime_type="application/pdf",
                status="indexed",
            ))
        await session.commit()


async def main(pdf_path: str) -> None:
    path = Path(pdf_path)
    if not path.exists():
        print(f"[错误] 文件不存在: {path}")
        sys.exit(1)

    filename = path.name
    t0 = time.time()

    # 1. 解析 + 清洗（B 阶段新增：去页脚/导航/URL/碎片行）
    pages = parse_pdf(path)
    pages = clean_pages(pages)
    print(f"[1/4] 解析+清洗完成：{len(pages)} 页有文本")

    # 2. 切分
    chunks = chunk_pages(pages, chunk_size=500, overlap=50)
    print(f"[2/4] 切分完成：{len(chunks)} 块（chunk_size=500, overlap=50）")

    # 3. 向量化（首次运行会下载模型，请耐心）
    embedder = get_embedder()
    print(f"[3/4] 向量化中...（模型 {embedder.model_name}，维度 {embedder.dim}）")
    vectors = embedder.embed_documents([c.text for c in chunks])

    # 4. 存 Qdrant（同名文件先清旧向量，再插入）
    client = get_client()
    ensure_collection(client, dim=embedder.dim)
    delete_points_of_file(client, filename)
    count = upsert_chunks(client, chunks, vectors, filename)

    # 5. documents 表登记
    await save_document_record(filename, str(path), path.stat().st_size, count)

    dt = time.time() - t0
    print(f"[4/4] Qdrant 已写入 {count} 个点，documents 表已登记")
    print(f"完成：耗时 {dt:.1f}s，平均 {dt / max(count, 1) * 1000:.0f}ms/块")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/langchain_intro.pdf"
    asyncio.run(main(target))
