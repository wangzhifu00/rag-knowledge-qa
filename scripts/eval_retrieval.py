"""检索质量评估脚本：B 阶段消融实验的量尺。

用法（项目根目录）：
    python scripts/eval_retrieval.py                        # baseline（不清洗, 500/50）
    python scripts/eval_retrieval.py --clean                # 清洗后
    python scripts/eval_retrieval.py --clean --chunk-size 300 --overlap 30

做法：用固定的 8 个测试问题，把文档按指定参数重建成一个临时集合
（eval_ 前缀，不碰正式的 documents_chunks），逐题检索后报：
  top1 均分 / top5 均分 —— 越高说明"问题向量和正确原文靠得越近"
跑完自动删除临时集合。
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient, models

from rag.chunker import chunk_pages
from rag.cleaner import clean_pages
from rag.embedder import get_embedder
from rag.parser import parse_pdf
from rag.retriever import get_client

PDF = "data/langchain_intro.pdf"
EVAL_COLLECTION = "eval_tmp"

# 固定测试问题集（8 个，覆盖概念/组件/操作/对比四类问法）
QUESTIONS = [
    "什么是LangChain",
    "LangChain提供哪些模块",
    "如何用LangChain做RAG",
    "OpenAIEmbeddings怎么用",
    "文本切分器是什么",
    "DeepAgents是什么",
    "怎么创建智能体",
    "支持哪些向量数据库",
]


def build_index(client: QdrantClient, embedder, chunk_size: int, overlap: int,
                clean: bool) -> int:
    pages = parse_pdf(PDF)
    if clean:
        pages = clean_pages(pages)
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    vectors = embedder.embed_documents([c.text for c in chunks])

    client.recreate_collection(
        collection_name=EVAL_COLLECTION,
        vectors_config=models.VectorParams(
            size=embedder.dim, distance=models.Distance.COSINE
        ),
    )
    upsert_chunks_eval(client, chunks, vectors)
    return len(chunks)


def upsert_chunks_eval(client: QdrantClient, chunks, vectors) -> None:
    """评估专用的简化写入（固定集合名，不需要 filename 过滤）。"""
    points = [
        models.PointStruct(
            id=c.chunk_index,
            vector=v,
            payload={"text": c.text, "page_numbers": c.page_numbers},
        )
        for c, v in zip(chunks, vectors)
    ]
    client.upsert(collection_name=EVAL_COLLECTION, points=points)


def evaluate(client: QdrantClient, embedder) -> dict:
    top1_scores, top5_scores = [], []
    nav_junk = 0    # 导航/页脚/碎片噪声（真正该清掉的）
    code_lines = 0  # import/常量等代码行（教程正文，合法内容）
    nav_re = re.compile(r"^(https?://\S+\s*)+$", re.M)
    code_re = re.compile(r"^import |^from .+ import|DOCS_BASE", re.M)
    footer_kw = ("changelog", "forum", "subscribe", "was this page", "interrupt is coming")
    for q in QUESTIONS:
        hits = client.query_points(
            collection_name=EVAL_COLLECTION,
            query=embedder.embed_query(q),
            limit=5,
        ).points
        scores = [p.score for p in hits]
        top1_scores.append(scores[0])
        top5_scores.append(sum(scores) / len(scores))
        for p in hits:
            for ln in p.payload.get("text", "").splitlines():
                s = ln.strip()
                if nav_re.search(s) or len(s) < 6 or (
                    len(s) < 80 and any(k in s.lower() for k in footer_kw)
                ):
                    nav_junk += 1
                if code_re.search(s):
                    code_lines += 1
    return {
        "avg_top1": sum(top1_scores) / len(top1_scores),
        "avg_top5": sum(top5_scores) / len(top5_scores),
        "nav_junk": nav_junk,
        "code_lines": code_lines,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--clean", action="store_true", help="启用文本清洗")
    args = ap.parse_args()

    client = get_client()
    embedder = get_embedder()

    n = build_index(client, embedder, args.chunk_size, args.overlap, args.clean)
    result = evaluate(client, embedder)

    mode = "清洗+重切" if args.clean else "baseline"
    print(f"[{mode}] chunk_size={args.chunk_size} overlap={args.overlap} → {n} 块 | "
          f"avg_top1={result['avg_top1']:.4f} | avg_top5={result['avg_top5']:.4f} | "
          f"40条结果中: 导航噪声 {result['nav_junk']} 条, 代码行 {result['code_lines']} 条")

    # 临时集合用完即删（正式的 documents_chunks 不受影响）
    client.delete_collection(EVAL_COLLECTION)


if __name__ == "__main__":
    main()
