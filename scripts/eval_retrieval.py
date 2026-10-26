"""检索质量评估：基于 golden 集（人工筛过的标注题）。

这是第⑦步的"尺子"——B 阶段的消融实验量的是"余弦分高低"，
本脚本量的是"该找到的找到没 / 不该编的有没有拒"。

题集从哪来（按优先级）：
  1. eval/golden_set.json —— 正式题集，由 scripts/gen_evalset.py 生成 + 人工筛
  2. 本文件内置的 10 题   —— 早期手写版，留作向后兼容的兜底
  存在 golden_set.json 时自动用它，打印时会写明来源，不用改命令。

用法（项目根目录）：
    python scripts/eval_retrieval.py                          # 三模式对比(默认 300/30，同正式集合)
    python scripts/eval_retrieval.py --mode both              # 只跑 向量 vs 混合
    python scripts/eval_retrieval.py --testset eval/golden_set.json  # 显式指题集
    python scripts/eval_retrieval.py --testset legacy         # 强制用内置 10 题
    python scripts/eval_retrieval.py --chunk-size 500 --overlap 50   # 换切分参数复现旧实验
    python scripts/eval_retrieval.py --no-clean               # 不清洗(baseline)

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
import json
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
GOLDEN_JSON = Path("eval/golden_set.json")

# golden 标注集：pages=标准答案所在页（命中判定=检索结果的页码与之相交）
#           pages=None 表示"库里没有答案"，正确行为是被拒答
# 标注方法：关键词全文定位（2026-10-26 机标，用户已核对）
# 注：这是早期手写版，只在 eval/golden_set.json 不存在时用（--testset legacy 也可强制）
LEGACY_TESTSET = [
    {"q": "什么是LangChain",         "pages": None},       # 全文未介绍 LangChain 本身
    {"q": "LangChain提供哪些模块",    "pages": None},       # 无模块列表章节
    {"q": "如何用LangChain做RAG",    "pages": [1]},        # RAG patterns 介绍页
    {"q": "OpenAIEmbeddings怎么用",  "pages": [10]},       # embeddings 用法 + VectorStore
    {"q": "文本切分器是什么",         "pages": [9]},        # Split documents 章节
    # 原标 pages=[1]（首页 intro），2026-10-26 复核后改判应拒答：
    # 全文只有 URL 路径 /deepagents/rag、代码里的 create_deep_agent、
    # 导航链接（RAG patterns / Grading rubrics），没有一句说明"Deep Agents 是什么"。
    # 这条标注错误是严格定义规则(prompt 第3条a)跑出来的——模型拒答是对的
    {"q": "DeepAgents是什么",        "pages": None},
    {"q": "怎么创建智能体",           "pages": [12]},       # Build the agent 章节
    {"q": "支持哪些向量数据库",       "pages": [10]},       # 集成列表(In-memory/AstraDB/Chroma...)
    {"q": "怎么运行RAG agent",       "pages": [16]},       # Run the agent 章节
    {"q": "文档切好块之后存到哪",     "pages": [10, 11]},   # VectorStore 持久化
]

# 实际使用的题集（main 里按 --testset 决定），格式统一为
# {"q": 问题, "pages": 标准页 或 None}；JSON 里多出来的字段这里用不上，忽略
TESTSET = LEGACY_TESTSET
TESTSET_SOURCE = "内置 10 题（legacy）"


def load_testset(arg: str) -> tuple[list[dict], str]:
    """按 --testset 载入题集，返回 (题集, 来源说明)。

    默认 auto：有 eval/golden_set.json 就用它，否则退回内置题集。
    """
    if arg == "legacy":
        return LEGACY_TESTSET, "内置 10 题（legacy，手动指定）"

    path = GOLDEN_JSON if arg == "auto" else Path(arg)
    if not path.exists():
        if arg == "auto":
            return LEGACY_TESTSET, "内置 10 题（未找到 eval/golden_set.json）"
        raise SystemExit(f"题集文件不存在：{path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    items = [{"q": i["q"], "pages": i.get("pages")} for i in data["items"]]
    n_ans = sum(1 for i in items if i["pages"])
    return items, (f"{path}｜{len(items)} 题"
                   f"（{n_ans} 有答案 + {len(items) - n_ans} 应拒答）")


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
    """单题检索：返回 SearchHit 列表（三种模式统一返回格式）。"""
    if mode == "hybrid_rerank":
        return hybrid_search(question, embedder, top_k=5, rerank=True,
                             collection_name=EVAL_COLLECTION)
    if mode == "hybrid":
        return hybrid_search(question, embedder, top_k=5,
                             collection_name=EVAL_COLLECTION)
    return search(client, embedder, question, top_k=5,
                  collection_name=EVAL_COLLECTION)


def evaluate_mode(client: QdrantClient, embedder, mode: str) -> dict:
    """跑一种模式的全部题目，返回指标和逐题明细。

    拒答门槛分模式：hybrid_rerank 用 reranker 概率(区分度高,主门槛)，
    其余退回向量相似度门槛。
    """
    settings = get_settings()
    threshold = (settings.rerank_min_score if mode == "hybrid_rerank"
                 else settings.retrieval_min_score)
    stats = {"hit1": 0, "hit5": 0, "mrr": [], "refusal_ok": 0, "n_refusal": 0}
    details = []
    pos_scores, neg_scores = [], []  # 有答案题/无答案题的 top1 分（阈值校准用）

    for item in TESTSET:
        q, golden = item["q"], item["pages"]
        hits = run_one(client, embedder, q, mode)
        # rerank 模式下 hits[0].score 就是 rerank 概率；其余取向量原始分
        top1 = hits[0].score if (mode == "hybrid_rerank" and hits) \
            else best_vector_score(hits)

        if golden is None:
            # 应拒答题：top1 分低于门槛才算对
            refused = top1 < threshold
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
    ap.add_argument("--chunk-size", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=30)
    ap.add_argument("--clean", dest="clean", action="store_true", default=True)
    ap.add_argument("--no-clean", dest="clean", action="store_false")
    ap.add_argument("--mode",
                    choices=["vector", "hybrid", "hybrid_rerank", "both", "all"],
                    default="all",
                    help="all=三种模式对比(默认) both=前两种")
    ap.add_argument("--testset", default="auto",
                    help="auto=有 eval/golden_set.json 就用(默认)；"
                         "legacy=内置 10 题；也可直接给 json 路径")
    args = ap.parse_args()

    global TESTSET, TESTSET_SOURCE
    TESTSET, TESTSET_SOURCE = load_testset(args.testset)

    settings = get_settings()
    client = get_client()
    embedder = get_embedder()
    warmup_jieba()

    n = build_index(client, embedder, args.chunk_size, args.overlap, args.clean)
    label = "清洗" if args.clean else "不清洗"
    print(f"\n题集：{TESTSET_SOURCE}")
    print(f"索引：{label} chunk={args.chunk_size}/{args.overlap} → {n} 块 | "
          f"门槛: 向量{settings.retrieval_min_score} "
          f"/ rerank{settings.rerank_min_score}\n")

    mode_map = {
        "vector": ["vector"],
        "hybrid": ["hybrid"],
        "hybrid_rerank": ["hybrid_rerank"],
        "both": ["vector", "hybrid"],
        "all": ["vector", "hybrid", "hybrid_rerank"],
    }
    mode_names = {
        "vector": "纯向量",
        "hybrid": "混合(RRF)",
        "hybrid_rerank": "混合+精排(Rerank)",
    }
    summary = {}
    for mode in mode_map[args.mode]:
        r = evaluate_mode(client, embedder, mode)
        summary[mode] = r
        print(f"—— {mode_names[mode]} ——")
        for q, golden, tag, top1 in r["details"]:
            print(f"  {tag:<7} top1={top1:.3f}  答案页[{golden}]  {q}")
        print(f"  hit@1={r['hit1']:.0%}  hit@5={r['hit5']:.0%}  "
              f"MRR={r['mrr']:.3f}  拒答正确率={r['refusal']:.0%}\n")

    # 汇总对比表（≥2 种模式时）
    if len(summary) >= 2:
        cols = "  ".join(f"{mode_names[m]:>14}" for m in summary)
        print("=" * (24 + 17 * len(summary)))
        print(f"{'指标':<22}{cols}")
        print("-" * (24 + 17 * len(summary)))
        rows = [
            ("hit@1", lambda r: f"{r['hit1']:.0%}"),
            ("hit@5", lambda r: f"{r['hit5']:.0%}"),
            ("MRR", lambda r: f"{r['mrr']:.3f}"),
            ("拒答正确率", lambda r: f"{r['refusal']:.0%}"),
        ]
        for label_, fn in rows:
            vals = "  ".join(f"{fn(summary[m]):>14}" for m in summary)
            print(f"{label_:<22}{vals}")
        print("=" * (24 + 17 * len(summary)))

    # 阈值校准建议：每种模式各自报告正例最低分/负例最高分
    for m, r in summary.items():
        scale = "rerank概率" if m == "hybrid_rerank" else "余弦相似度"
        if r["neg_max"] < r["pos_min"]:
            mid = (r["pos_min"] + r["neg_max"]) / 2
            print(f"[{mode_names[m]}] 可分：正例最低={r['pos_min']:.3f} "
                  f"负例最高={r['neg_max']:.3f} → 建议门槛≈{mid:.2f} ({scale})")
        else:
            print(f"[{mode_names[m]}] ⚠️ 分数重叠(正例最低{r['pos_min']:.3f} ≤ "
                  f"负例最高{r['neg_max']:.3f})，门槛分不开，需 prompt 层兜底 ({scale})")

    client.delete_collection(EVAL_COLLECTION)


if __name__ == "__main__":
    main()
