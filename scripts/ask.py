"""问答脚本：走完 RAG 问答线。

用法（项目根目录）：
    python scripts/ask.py "什么是RAG"               # 纯向量检索
    python scripts/ask.py "什么是RAG" --hybrid      # 混合检索（向量+BM25+RRF）
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from rag.embedder import get_embedder
from rag.generator import format_sources, generate_answer
from rag.retriever import best_vector_score, get_client, search

settings = get_settings()


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--hybrid"]
    use_hybrid = "--hybrid" in sys.argv
    if not args:
        print('用法: python scripts/ask.py "你的问题" [--hybrid]')
        sys.exit(1)
    question = " ".join(args)

    t0 = time.time()

    # 1. 检索：默认纯向量；--hybrid 走 向量+BM25+RRF 融合
    embedder = get_embedder()
    if use_hybrid:
        from rag.hybrid import hybrid_search, warmup_jieba

        warmup_jieba()
        hits = hybrid_search(question, embedder, top_k=5)
        mode = "混合(向量+BM25+RRF)"
    else:
        client = get_client()
        hits = search(client, embedder, question, top_k=5)
        mode = "纯向量"
    t_search = time.time() - t0

    if not hits:
        print("知识库中检索不到任何内容。请先运行: python scripts/ingest.py")
        sys.exit(1)

    # 1.5 前置拒答：top1 向量相似度低于门槛 → 不调 LLM 直接拒。
    # 防止"库里没有的内容，模型自己编一段还标假引用"的幻觉
    top_score = best_vector_score(hits)
    if top_score < settings.retrieval_min_score:
        print("=" * 60)
        print(f"检索模式：{mode}")
        print("=" * 60)
        print("知识库中未找到与该问题足够相关的内容，无法根据现有资料回答。")
        print(f"(top1 向量相似度 {top_score:.3f} < 门槛 {settings.retrieval_min_score})")
        print(f"\n耗时：检索 {t_search:.2f}s（未调用 LLM）")
        return

    # 2. 生成：原文 + 问题 → LLM
    t1 = time.time()
    answer = generate_answer(question, hits)
    t_gen = time.time() - t1

    print("=" * 60)
    print(f"检索模式：{mode}")
    print("=" * 60)
    print(answer)
    print("=" * 60)
    print("来源：")
    print(format_sources(hits))
    print(f"\n耗时：检索 {t_search:.2f}s + 生成 {t_gen:.2f}s")


if __name__ == "__main__":
    main()
