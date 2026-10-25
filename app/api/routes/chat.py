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

from app.config import get_settings
from rag.embedder import get_embedder
from rag.generator import generate_answer
from rag.retriever import best_vector_score, get_client, search

settings = get_settings()

router = APIRouter(tags=["chat"])

# 检索质量不达标时的统一拒答话术（不调 LLM，直接返回）
REFUSAL_ANSWER = "知识库中未找到与该问题足够相关的内容，无法根据现有资料回答。"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20, description="检索条数")
    mode: str = Field(
        default="hybrid",
        pattern="^(vector|hybrid)$",
        description="检索模式：vector=纯向量, hybrid=向量+BM25+RRF 融合",
    )


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
    mode: str
    refused: bool = False          # True=触发了检索质量门槛拒答
    top_vector_score: float = 0.0  # top1 向量原始相似度（调试/评估用）
    elapsed: dict


@router.post("/chat", response_model=ChatResponse, summary="知识库问答（RAG）")
def chat(req: ChatRequest) -> ChatResponse:
    t0 = time.time()

    # 检索（同步阻塞，线程池里跑）。默认 hybrid，想对比可传 mode=vector
    embedder = get_embedder()
    if req.mode == "hybrid":
        from rag.hybrid import hybrid_search, warmup_jieba

        warmup_jieba()
        hits = hybrid_search(req.question, embedder, top_k=req.top_k)
    else:
        client = get_client()
        hits = search(client, embedder, req.question, top_k=req.top_k)
    t_search = time.time() - t0

    # 前置拒答：top1 向量相似度低于门槛 → 不调 LLM 直接拒。
    # 背景：曾出现"库里没有的内容，模型自己编一段还标了假引用"——
    # 与其让 LLM 硬编，不如在检索层就承认"库里没有"。
    top_score = best_vector_score(hits)
    elapsed_partial = {
        "search_seconds": round(t_search, 2),
        "total_seconds": round(time.time() - t0, 2),
    }
    if top_score < settings.retrieval_min_score:
        return ChatResponse(
            answer=REFUSAL_ANSWER,
            sources=[],
            question=req.question,
            mode=req.mode,
            refused=True,
            top_vector_score=round(top_score, 4),
            elapsed=elapsed_partial,
        )

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
        mode=req.mode,
        refused=False,
        top_vector_score=round(top_score, 4),
        elapsed={
            "search_seconds": round(t_search, 2),
            "generate_seconds": round(t_generate, 2),
            "total_seconds": round(time.time() - t0, 2),
        },
    )
