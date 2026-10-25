"""文档管理路由：上传 PDF → 后台入库，查看入库状态。

为什么用 BackgroundTasks：大 PDF 向量化要几十秒，不能让用户的 HTTP 请求
一直挂着。做法是：立即返回"已接收"，真正的解析/切分/向量化在后台跑，
进度看 documents 表的 status 字段（uploaded → parsing → indexed / failed）。
这就是路线第 6 周补充里"文档上传后异步处理"的最小实现。
"""

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_db
from app.db.models import Document
from rag.chunker import chunk_pages
from rag.embedder import get_embedder
from rag.parser import parse_pdf
from rag.retriever import (
    delete_points_of_file,
    ensure_collection,
    get_client,
    upsert_chunks,
)

router = APIRouter(tags=["documents"])
logger = logging.getLogger(__name__)

# 上传文件保存目录：项目根/data/uploads
UPLOAD_DIR = Path(__file__).resolve().parents[2] / "data" / "uploads"


async def _set_status(doc_id: int, status: str, error: str | None = None) -> None:
    """独立小函数：更新一条记录的状态（后台任务用自己的 session）。"""
    from app.db.engine import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, doc_id)
        if doc:
            doc.status = status
            doc.error_message = error
            await session.commit()


async def process_document(doc_id: int, file_path: Path, filename: str) -> None:
    """后台入库任务：解析 → 切分 → 向量化 → 存 Qdrant → 更新状态。

    注意：pypdf/fastembed/Qdrant 客户端都是同步阻塞的，
    用 asyncio.to_thread 包一层丢进线程池，避免卡死事件循环。
    任何一环失败都把 status 置为 failed 并记录原因，不静默吞掉。
    """
    try:
        await _set_status(doc_id, "parsing")

        pages = await asyncio.to_thread(parse_pdf, file_path)
        chunks = await asyncio.to_thread(chunk_pages, pages)

        await _set_status(doc_id, "embedding")

        def _embed_and_store() -> int:
            embedder = get_embedder()
            vectors = embedder.embed_documents([c.text for c in chunks])
            client = get_client()
            ensure_collection(client, dim=embedder.dim)
            delete_points_of_file(client, filename)  # 重复上传先清旧向量
            return upsert_chunks(client, chunks, vectors, filename)

        count = await asyncio.to_thread(_embed_and_store)
        logger.info("文档 %s 入库完成：%d 块", filename, count)
        await _set_status(doc_id, "indexed")

    except Exception as exc:  # noqa: BLE001
        logger.exception("文档 %s 入库失败", filename)
        await _set_status(doc_id, "failed", str(exc)[:500])


@router.post("/documents/upload", summary="上传 PDF 文档（异步入库）")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
) -> dict:
    # 基本校验：只收 PDF
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOAD_DIR / file.filename  # 同名覆盖，和检索层的去重逻辑一致
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件为空")
    save_path.write_bytes(content)

    # 先登记一条 uploaded 记录，立刻返回；入库在后台进行
    result = await db.execute(
        select(Document).where(Document.filename == file.filename)
    )
    doc = result.scalar_one_or_none()
    if doc:
        doc.file_path = str(save_path)
        doc.file_size = len(content)
        doc.status = "uploaded"
        doc.error_message = None
    else:
        doc = Document(
            filename=file.filename,
            file_path=str(save_path),
            file_size=len(content),
            mime_type=file.content_type or "application/pdf",
            status="uploaded",
        )
        db.add(doc)
    await db.commit()
    await db.refresh(doc)

    background_tasks.add_task(process_document, doc.id, save_path, file.filename)

    return {
        "id": doc.id,
        "filename": doc.filename,
        "status": doc.status,
        "message": "已接收，正在后台解析入库。稍后用 GET /documents 查看状态",
    }


@router.get("/documents", summary="文档列表（含入库状态）")
async def list_documents(db: AsyncSession = Depends(get_db)) -> list[dict]:
    result = await db.execute(select(Document).order_by(Document.id))
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "file_size": d.file_size,
            "status": d.status,
            "error_message": d.error_message,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in result.scalars()
    ]
