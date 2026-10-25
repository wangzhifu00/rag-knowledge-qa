"""检索质量评估：基于人工标注的 golden 集（10 题）。

这是第⑦步的"尺子"——B 阶段的消融实验量的是"余弦分高低"，
本脚本量的是"该找到的找到没 / 不该编的有没有拒"。

用法（项目根目录）：
    python scripts/eval_retrieval.py                                  # 双模式对比(默认 500/50 清洗)
    python scripts/eval_retrieval.py --chunk-size 300 --overlap 30    # 换切分参数
    python scripts/eval_retrieval.py --no-clean                       # 不清洗(baseline)

指标定义：
  hit@1      第一条就命中标准页的比例（answerable 题）
  hit@5      前 5 条里含标准页的比例（answerable 题）
  MRR        首个命中位置的倒数平均，反映"排得靠不靠前"
  拒答正确率  库里没答案的题，top1 相似度低于门槛被判拒的比例

做法：按指定参数把文档重建成临时集合（eval_tmp，不碰正式集合），
向量/混合两种模式各跑一遍全部题目，最后给出对比表和阈值校准建议。
跑完自动删除临时集合。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient, models

from app.config import get_settings
from rag.chunker import chunk_pages
from rag.cleaner import clean_pages
from rag.embedder import get_embedder
from rag.hybrid import hybrid_search, warmup_jieba
from rag.parser import parse_pdf
from rag.retriever import best_vector_score, get_client, search

PDF = "data/langchain_intro.pdf"
EVAL_COLLECTION = "eval_tmp"

# golden 标注集：pages=标准答案所在页（命中判定=检索结果的页码与之相交）
#           pages=None 表示"库里没有答案"，正确行为是被拒答
# 标注方法：关键词全文定位（2026-10-26 机标，用户已核对）
TESTSET = [
    {"q": "什么是LangChain",         "pages": None},       # 全文未介绍 LangChain 本身
    {"q": "LangChain提供哪些模块",    "pages": None},       # 无模块列表章节
    {"q": "如何用LangChain做RAG",    "pages": [1]},        # RAG patterns 介绍页
    {"q": "OpenAIEmbeddings怎么用",  "pages": [10]},       # embeddings 用法 + VectorStore
    {"q": "文本切分器是什么",         "pages": [9]},        # Split documents 章节
    {"q": "DeepAgents是什么",        "pages": [1]},        # 首页 intro
    {"q": "怎么创建智能体",           "pages": [12]},       # Build the agent 章节
    {"q": "支持哪些向量数据库",       "pages": [10]},       # 集成列表(In-memory/AstraDB/Chroma...)
    {"q": "怎么运行RAG agent",       "pages": [16]},       # Run the agent 章节
    {"q": "文档切好块之后存到哪",     "pages": [10, 11]},   # VectorStore 持久化
]


def build_index(client: QdrantClient, embedder, chunk_size: int, overlap: int,
                clean: bool) -> int:
    """按指定参数重建临时评估集合，返回块数。"""
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
    points = [
        models.PointStruct(
            id=c.chunk_index,
            vector=v,
            payload={
                "text": c.text,
                "filename": "eval.pdf",
                "page_numbers": c.page_numbers,
                "chunk_index": c.chunk_index,
            },
        )
        for c, v in zip(chunks, vectors)
    ]
    client.upsert(collection_name=EVAL_COLLECTION, points=points)
    return len(chunks)


def run_one(client: QdrantClient, embedder, question: str, mode: str) -> list:
    """单题检索：返回 SearchHit 列表（两种模式统一返回格式）。"""
    if mode == "hybrid":
        return hybrid_search(question, embedder, top_k=5,
                             collection_name=EVAL_COLLECTION)
    return search(client, embedder, question, top_k=5,
                  collection_name=EVAL_COLLECTION)


def evaluate_mode(client: QdrantClient, embedder, mode: str, min_score: float) -> dict:
    """跑一种模式的全部题目，返回指标和逐题明细。"""
    stats = {"hit1": 0, "hit5": 0, "mrr": [], "refusal_ok": 0, "n_refusal": 0}
    details = []
    pos_scores, neg_scores = [], []  # 有答案题/无答案题的 top1 分（阈值校准用）

    for item in TESTSET:
        q, golden = item["q"], item["pages"]
        hits = run_one(client, embedder, q, mode)
        top1 = best_vector_score(hits)

        if golden is None:
            # 应拒答题：top1 分低于门槛才算对
            refused = top1 < min_score
            stats["n_refusal"] += 1
            stats["refusal_ok"] += int(refused)
            neg_scores.append(top1)
            details.append((q, "应拒答", "✅拒" if refused else "❌未拒", top1))
            continue

        # 有答案题：命中判定 = 结果页码与标准页相交
        rank = next((i for i, h in enumerate(hits, 1)
                     if set(h.page_numbers) & set(golden)), 0)
        hit1 = rank == 1
        hit5 = rank > 0
        stats["hit1"] += int(hit1)
        stats["hit5"] += int(hit5)
        stats["mrr"].append(1.0 / rank if rank else 0.0)
        pos_scores.append(top1)
        tag = f"第{rank}位" if rank else "未命中"
        details.append((q, "/".join(map(str, golden)), tag, top1))

    n_ans = len(TESTSET) - stats["n_refusal"]
    return {
        "details": details,
        "hit1": stats["hit1"] / n_ans,
        "hit5": stats["hit5"] / n_ans,
        "mrr": sum(stats["mrr"]) / n_ans,
        "refusal": stats["refusal_ok"] / stats["n_refusal"],
        "pos_min": min(pos_scores),
        "neg_max": max(neg_scores),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--clean", dest="clean", action="store_true", default=True)
    ap.add_argument("--no-clean", dest="clean", action="store_false")
    ap.add_argument("--mode", choices=["vector", "hybrid", "both"], default="both")
    args = ap.parse_args()

    settings = get_settings()
    client = get_client()
    embedder = get_embedder()
    warmup_jieba()

    n = build_index(client, embedder, args.chunk_size, args.overlap, args.clean)
    label = "清洗" if args.clean else "不清洗"
    print(f"\n评估集：{label} chunk={args.chunk_size}/{args.overlap} → {n} 块 | "
          f"10 题（8 有答案 + 2 应拒答）| 拒答门槛 {settings.retrieval_min_score}\n")

    modes = ["vector", "hybrid"] if args.mode == "both" else [args.mode]
    summary = {}
    for mode in modes:
        r = evaluate_mode(client, embedder, mode, settings.retrieval_min_score)
        summary[mode] = r
        name = "纯向量" if mode == "vector" else "混合(RRF)"
        print(f"—— {name} ——")
        for q, golden, tag, top1 in r["details"]:
            print(f"  {tag:<7} top1={top1:.3f}  答案页[{golden}]  {q}")
        print(f"  hit@1={r['hit1']:.0%}  hit@5={r['hit5']:.0%}  "
              f"MRR={r['mrr']:.3f}  拒答正确率={r['refusal']:.0%}\n")

    if len(summary) == 2:
        v, h = summary["vector"], summary["hybrid"]
        print("=" * 56)
        print(f"对比：hit@5 {v['hit5']:.0%} → {h['hit5']:.0%} | "
              f"MRR {v['mrr']:.3f} → {h['mrr']:.3f} | "
              f"拒答 {v['refusal']:.0%} → {h['refusal']:.0%}")
        print("=" * 56)

    # 阈值校准建议：正例最低分与负例最高分之间取中点
    pos = min(summary[m]["pos_min"] for m in summary)
    neg = max(summary[m]["neg_max"] for m in summary)
    if neg < pos:
        print(f"阈值校准：正例最低 top1={pos:.3f}，负例最高 top1={neg:.3f}，"
              f"建议门槛 ≈ {(pos + neg) / 2:.2f}")
    else:
        print(f"⚠️ 分数区间重叠（负例最高 {neg:.3f} ≥ 正例最低 {pos:.3f}），"
              f"单靠分数门槛分不开，需保留 prompt 层兜底")

    client.delete_collection(EVAL_COLLECTION)


if __name__ == "__main__":
    main()
