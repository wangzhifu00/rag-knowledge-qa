"""问答路由：RAG 问答线包成 API。

两个接口，一套检索逻辑：
  POST /chat         一次性返回完整 JSON（脚本 / 评估脚本 / 其他系统调用）
  POST /chat/stream  SSE 流式返回（网页前端用，打字机效果）

为什么流式用 SSE 而不是 WebSocket：
这条链路是"客户端问一次、服务端往下推"，单向、没心跳、没重连需求。
SSE 本质就是一个不关闭的 HTTP 响应，浏览器用 fetch 读 body 流就能消费，
不用引 WebSocket 那套连接管理。反过来，如果要做多人协作/双向推送才上 WS。

顺带保留了原同步路由的注释：同步 def 路由 FastAPI 会自动丢线程池，
不会阻塞事件循环——比手动 asyncio.to_thread 更省事。
"""

import asyncio
import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from rag.embedder import get_embedder
from rag.generator import generate_answer, stream_answer
from rag.retriever import SearchHit, best_vector_score, get_client, search
from rag.smalltalk import is_smalltalk, smalltalk_answer

settings = get_settings()

router = APIRouter(tags=["chat"])

# 检索质量不达标时的统一拒答话术（不调 LLM，直接返回）
REFUSAL_ANSWER = "知识库中未找到与该问题足够相关的内容，无法根据现有资料回答。"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20, description="检索条数")
    mode: str = Field(
        default="hybrid_rerank",  # 评估胜出配置: hit@5 88% / MRR 0.72
        pattern="^(vector|hybrid|hybrid_rerank)$",
        description="检索模式：vector=纯向量, hybrid=向量+BM25+RRF, "
                    "hybrid_rerank=再加交叉编码器精排(多花1~2秒)",
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
    refused: bool = False          # True=模型按 prompt 判定材料不足（以固定开头识别）
    smalltalk: bool = False        # True=闲聊/身份类，未检索
    below_threshold: bool = False  # True=检索分低于门槛，材料已丢弃（可能是通用问题）
    top_vector_score: float = 0.0  # top1 分数（调试/评估用）
    elapsed: dict


# ---------------------------------------------------------------- 共享逻辑
# 两个接口都要"检索 + 判拒答 + 组来源"，抽出来避免复制粘贴两遍。
# 以后改检索策略只动这一个函数，两个接口同时生效。

def _retrieve(question: str, top_k: int, mode: str) -> tuple[list[SearchHit], float]:
    """检索并返回 (hits, 用于拒答判断的 top 分数)。

    纯同步阻塞函数（embedding / jieba / reranker 全是 CPU 计算），
    调用方负责丢进线程池，别在事件循环里直接跑。
    """
    embedder = get_embedder()

    if mode in ("hybrid", "hybrid_rerank"):
        from rag.hybrid import hybrid_search, warmup_jieba

        warmup_jieba()  # 首次分词要加载词典，预热一下免得第一问特别慢
        hits = hybrid_search(
            question, embedder, top_k=top_k, rerank=(mode == "hybrid_rerank")
        )
    else:
        client = get_client()
        hits = search(client, embedder, question, top_k=top_k)

    # 拒答打分模式——rerank 模式用 reranker 概率(区分度高，是主门槛)，
    # 其余模式退回向量相似度门槛(区分度不足，仅兜底)。见 _refusal_threshold。
    if mode == "hybrid_rerank":
        return hits, max((h.score for h in hits), default=0.0)
    return hits, best_vector_score(hits)


def _refusal_threshold(mode: str) -> float:
    if mode == "hybrid_rerank":
        return settings.rerank_min_score
    return settings.retrieval_min_score


def _to_sources(hits: list[SearchHit]) -> list[SourceOut]:
    return [
        SourceOut(
            index=i,
            filename=h.filename,
            pages=h.page_numbers,
            score=round(h.score, 4),
            preview=h.text[:160].replace("\n", " "),
        )
        for i, h in enumerate(hits, start=1)
    ]


# ---------------------------------------------------------------- 同步接口
@router.post("/chat", response_model=ChatResponse, summary="知识库问答（RAG）")
def chat(req: ChatRequest) -> ChatResponse:
    t0 = time.time()

    # 闲聊/身份类短路：不检索、不调模型。
    # 这类问题检索必然命中一堆无关块，右栏来源面板会误导用户
    # （实测问"你是什么"，返回 5 个 0.588 分的代码片段）；
    # 而且白跑一次 embedding+rerank。判定见 rag/smalltalk.py。
    if is_smalltalk(req.question):
        return ChatResponse(
            answer=smalltalk_answer(req.question),
            sources=[],
            question=req.question,
            mode=req.mode,
            smalltalk=True,
            elapsed={"search_seconds": 0.0, "generate_seconds": 0.0,
                     "total_seconds": round(time.time() - t0, 2)},
        )

    hits, top_score = _retrieve(req.question, req.top_k, req.mode)
    t_search = time.time() - t0

    # 分数低于门槛：把命中结果丢掉（低分块全是噪声，喂给模型只会干扰），
    # 但**不再直接返回拒答**——材料为空有两种可能：
    #   ① 这题库里真没有 → 模型按规则 3-a / 3-b 拒答
    #   ② 这根本不是知识库范围的问题（"Python 怎么读文件"）→ 规则 3-c 用通用知识答
    # 以前在检索层就把 ② 一起拒掉了，等于把模型的通用能力阉了。
    below_threshold = top_score < _refusal_threshold(req.mode)
    if below_threshold:
        hits = []

    t1 = time.time()
    answer = generate_answer(req.question, hits) or REFUSAL_ANSWER
    t_generate = time.time() - t1

    # 拒答由模型自己决定（按 prompt 的固定开头判断），后端只做识别
    refused = answer.lstrip().startswith("根据现有资料无法回答")

    return ChatResponse(
        answer=answer,
        sources=[] if refused else _to_sources(hits),
        question=req.question,
        mode=req.mode,
        refused=refused,
        below_threshold=below_threshold,
        top_vector_score=round(top_score, 4),
        elapsed={
            "search_seconds": round(t_search, 2),
            "generate_seconds": round(t_generate, 2),
            "total_seconds": round(time.time() - t0, 2),
        },
    )


# ---------------------------------------------------------------- 流式接口
def _sse(payload: dict) -> str:
    """SSE 协议格式：`data: <json>` + 两个换行。

    两个换行是消息分隔符（不是排版习惯），前端按 "\\n\\n" 切包。
    ensure_ascii=False 让中文原样输出，不然全是 \\uXXXX，调试时看不懂。
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat/stream", summary="知识库问答（SSE 流式）")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """事件序列（前端按 type 分发）：

    meta    开场，带上检索分数和门槛，前端可以显示"检索耗时 0.8s"
    sources 来源列表，先推——这样用户看字的同时右边来源已经在了
    token   增量文本，重复 N 次
    done    收尾，带各阶段耗时
    error   出错（检索或生成），前端显示红条

    拒答时不出 token 流，直接把整句话当一个 token 推完，前端逻辑统一。
    """

    async def event_gen():
        t0 = time.time()

        # 闲聊/身份类短路：顶在最前面，一次正则匹配就分流。
        # top_score 给 None，前端据此显示"未检索"，不显示 top/门槛。
        if is_smalltalk(req.question):
            yield _sse({
                "type": "meta", "question": req.question, "mode": req.mode,
                "top_score": None, "threshold": None,
                "search_seconds": 0.0, "smalltalk": True,
            })
            yield _sse({"type": "sources", "sources": []})
            yield _sse({"type": "token", "text": smalltalk_answer(req.question)})
            yield _sse({
                "type": "done", "refused": False, "smalltalk": True,
                "elapsed": {"search_seconds": 0.0, "generate_seconds": 0.0,
                            "total_seconds": round(time.time() - t0, 2)},
            })
            return

        # 检索丢线程池：embedding + reranker 是 CPU 密集的，
        # 直接在事件循环里算会把整条流卡住（这时候别的请求全在排队）。
        try:
            hits, top_score = await asyncio.to_thread(
                _retrieve, req.question, req.top_k, req.mode
            )
        except Exception as exc:  # noqa: BLE001
            yield _sse({"type": "error", "message": f"检索失败：{exc}"})
            return

        t_search = time.time() - t0
        threshold = _refusal_threshold(req.mode)
        # 低分时丢弃命中块（全是噪声），但继续调模型——由 prompt 判断
        # 是"库里有主题没内容"（拒答）还是"根本不属于知识库"（通用知识回答）
        below = top_score < threshold
        if below:
            hits = []

        yield _sse({
            "type": "meta",
            "question": req.question,
            "mode": req.mode,
            "top_score": round(top_score, 4),
            "threshold": threshold,
            "search_seconds": round(t_search, 2),
            "below_threshold": below,
        })

        yield _sse({
            "type": "sources",
            "sources": [] if below else [s.model_dump() for s in _to_sources(hits)],
        })

        t1 = time.time()
        acc: list[str] = []
        try:
            async for piece in stream_answer(req.question, hits):
                acc.append(piece)
                yield _sse({"type": "token", "text": piece})
        except Exception as exc:  # noqa: BLE001
            yield _sse({"type": "error", "message": f"生成失败：{exc}"})
            return

        # 拒答由模型自己决定（固定开头），后端只识别——不再由检索门槛拍板
        answer = "".join(acc)
        refused = answer.lstrip().startswith("根据现有资料无法回答")

        yield _sse({
            "type": "done",
            "refused": refused,
            "below_threshold": below,
            "elapsed": {
                "search_seconds": round(t_search, 2),
                "generate_seconds": round(time.time() - t1, 2),
                "total_seconds": round(time.time() - t0, 2),
            },
        })

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # nginx 默认会把响应攒满 buffer 再发，流式就被"憋"成一整块。
            # 本地直连 uvicorn 不受影响，这条是给以后上 nginx 留的。
            "X-Accel-Buffering": "no",
        },
    )
