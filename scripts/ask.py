"""问答脚本：走完 RAG 问答线。

用法（项目根目录）：
    python scripts/ask.py "什么是RAG"
    python scripts/ask.py "LangChain 的核心组件有哪些"
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.embedder import get_embedder
from rag.generator import format_sources, generate_answer
from rag.retriever import get_client, search


def main() -> None:
    if len(sys.argv) < 2:
        print('用法: python scripts/ask.py "你的问题"')
        sys.exit(1)
    question = " ".join(sys.argv[1:])

    t0 = time.time()

    # 1. 检索：问题 → 向量 → Qdrant 找最近 top5
    embedder = get_embedder()
    client = get_client()
    hits = search(client, embedder, question, top_k=5)
    t_search = time.time() - t0

    if not hits:
        print("知识库中检索不到任何内容。请先运行: python scripts/ingest.py")
        sys.exit(1)

    # 2. 生成：原文 + 问题 → DeepSeek
    t1 = time.time()
    answer = generate_answer(question, hits)
    t_gen = time.time() - t1

    print("=" * 60)
    print(answer)
    print("=" * 60)
    print("来源：")
    print(format_sources(hits))
    print(f"\n耗时：检索 {t_search:.2f}s + 生成 {t_gen:.2f}s")


if __name__ == "__main__":
    main()
