"""问答路由：RAG 问答线包成 API。

POST /chat  {"question": "...", "top_k": 5}
返回结构化 JSON：answer + sources 数组（文件名/页码/相似度），
前端渲染"点击引用跳原文"靠的就是这个结构。

这里故意用同步 def 路由：FastAPI 会自动把同步路由丢进线程池跑，
不会阻塞事件循环——比手动 asyncio.to_thread 更省事。
"""

import time

from fastapi import APIRouter
from pydantic import BaseModel, Field

from rag.embedder import get_embedder
from rag.generator import generate_answer
from rag.retriever import get_client, search

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20, description="检索条数")


class SourceOut(BaseModel):
    index: int
    filename: str
    pages: list[int]
    score: float
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    question: str
    elapsed: dict


@router.post("/chat", response_model=ChatResponse, summary="知识库问答（RAG）")
def chat(req: ChatRequest) -> ChatResponse:
    t0 = time.time()

    # 检索（同步阻塞，线程池里跑）
    embedder = get_embedder()
    client = get_client()
    hits = search(client, embedder, req.question, top_k=req.top_k)
    t_search = time.time() - t0

    # 生成
    t1 = time.time()
    answer = generate_answer(req.question, hits)
    t_generate = time.time() - t1

    sources = [
        SourceOut(
            index=i,
            filename=h.filename,
            pages=h.page_numbers,
            score=round(h.score, 4),
            preview=h.text[:80].replace("\n", " "),
        )
        for i, h in enumerate(hits, start=1)
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        question=req.question,
        elapsed={
            "search_seconds": round(t_search, 2),
            "generate_seconds": round(t_generate, 2),
            "total_seconds": round(time.time() - t0, 2),
        },
    )
