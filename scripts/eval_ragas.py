"""RAGAS 答案质量评估：评估的"决赛"。

前面 eval_retrieval.py 量的是"检索找得准不准"（第一层，免费、秒出），
本脚本用 RAGAS 量"最终答案质量好不好"（第二层，裁判是 LLM，烧 API）：

  faithfulness       忠实度：答案是否都有检索材料支撑（反幻觉，最关键）
  answer_relevancy   切题度：答案是否对准了问题
  context_precision  检索精度：检回来的材料里，有用的排得靠不靠前
  context_recall     检索召回：该有的材料都检回来了吗

设计：
  - 只评 8 道有答案的题。应拒答题没有参考答案，RAGAS 打不了分，
    它们的表现已由第一层评估的"拒答正确率"覆盖
  - 答案不是人写的，是系统真实生成的（generate_answer），检索也走
    真实链路——评的就是线上行为本身
  - vector / hybrid 两种模式各跑一遍，对比哪个配置答案质量更高
  - 裁判 LLM 和评估用向量化都用 .env 里配的 DashScope（qwen）

用法（项目根目录）：
    python scripts/eval_ragas.py

成本提示：每个指标每题要调若干次裁判 LLM，全量约一两百次调用，
qwen-plus 计费下大约几毛钱，跑一次几分钟。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from rag.embedder import get_embedder
from rag.generator import generate_answer
from rag.hybrid import hybrid_search, warmup_jieba
from rag.retriever import best_vector_score, get_client, search

settings = get_settings()

# 评估用的 embedding 模型（DashScope 兼容接口提供的向量化服务）
# 切题度指标需要把答案转向量，用 API 服务省去本地装 torch
JUDGE_EMBEDDING_MODEL = "text-embedding-v3"

# 8 道有答案题的参考答案（依据文档原文人工提炼，供 RAGAS 对比打分）
REFERENCE = {
    "如何用LangChain做RAG":
        "按教程分两步：先索引文档（加载、切分、向量化、存入向量库），"
        "再构建带检索工具的智能体，回答时先检索相关块再综合生成。",
    "OpenAIEmbeddings怎么用":
        "使用 OpenAIEmbeddings 生成文本向量，配合 VectorStore 存储文档块"
        "及向量，供相似度检索使用。",
    "文本切分器是什么":
        "文本切分器把长文档拆成小块（如 RecursiveCharacterTextSplitter），"
        "便于索引和检索，也方便传给模型。",
    "DeepAgents是什么":
        "Deep Agents 是 LangChain 的智能体框架，可编排检索、分析、综合等任务，"
        "支持子代理、技能和评分细则（grading rubrics）。",
    "怎么创建智能体":
        "在 agent.py 中用 create_deep_agent 创建智能体：配置相似度检索工具、"
        "StateBackend 文件后端、编排工作流和子代理提示词。",
    "支持哪些向量数据库":
        "支持 In-memory、Amazon OpenSearch、AstraDB、Chroma、Milvus、"
        "MongoDB、PGVector 等多种向量库集成。",
    "怎么运行RAG agent":
        "运行 agent.py，调用 agent.invoke 传入示例问题；智能体会先检索、"
        "把命中的块写入文件系统，再综合生成回答。",
    "文档切好块之后存到哪":
        "存入 VectorStore（向量数据库），连同 embeddings 一起持久化，"
        "供提问时做相似度检索。",
}


def retrieve(question: str, mode: str) -> list:
    """走真实检索链路拿 top5（在正式集合 documents_chunks 上）。"""
    client = get_client()
    if mode == "hybrid":
        return hybrid_search(question, get_embedder(), top_k=5)
    return search(client, get_embedder(), question, top_k=5)


def collect_records(mode: str) -> tuple[list[dict], list[str]]:
    """生成一种模式下的 RAGAS 数据：真实检索 + 系统真实生成的答案。

    被拒答门槛拦下的问题会跳过并记录——RAGAS 没法给拒答话术
    打忠实度分，这类题的表现由第一层评估的拒答正确率覆盖。
    """
    records, skipped = [], []
    for q, reference in REFERENCE.items():
        hits = retrieve(q, mode)
        top1 = best_vector_score(hits)
        if top1 < settings.retrieval_min_score:
            skipped.append(f"{q}（top1={top1:.3f} 低于门槛，系统拒答）")
            continue
        answer = generate_answer(q, hits)
        records.append({
            "user_input": q,
            "retrieved_contexts": [h.text for h in hits],
            "response": answer,
            "reference": reference,
        })
        print(f"  [{mode}] 已生成: {q} (top1={top1:.3f})")
    return records, skipped


def main() -> None:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)

    # 裁判 LLM：就是业务用的 qwen（OpenAI 兼容协议，换 base_url 即可）
    judge_llm = LangchainLLMWrapper(ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,  # 裁判要稳，温度归零
    ))
    # 切题度指标需要向量化答案：用 DashScope 的 embedding 服务
    judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=JUDGE_EMBEDDING_MODEL,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        # 非官方 OpenAI 端点必须关掉本地的 token 截断逻辑
        check_embedding_ctx_length=False,
    ))

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    warmup_jieba()

    results = {}
    for mode in ("vector", "hybrid"):
        print(f"\n===== [{mode}] 生成答案 =====")
        records, skipped = collect_records(mode)
        for s in skipped:
            print(f"  跳过: {s}")
        if not records:
            print(f"  [{mode}] 没有可评估的样本，跳过")
            continue

        print(f"===== [{mode}] RAGAS 打分中（裁判LLM多次调用，耐心等）=====")
        t0 = time.time()
        result = evaluate(
            EvaluationDataset.from_list(records),
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
        )
        df = result.to_pandas()
        results[mode] = df
        print(f"  ({time.time() - t0:.0f}s)\n")
        cols = ["user_input", "faithfulness", "answer_relevancy",
                "context_precision", "context_recall"]
        print(df[cols].round(3).to_string(index=False))

    # 汇总对比：两个模式各指标的平均分
    print("\n" + "=" * 62)
    print(f"{'指标':<20}{'纯向量':>10}{'混合(RRF)':>12}")
    print("-" * 62)
    for m in ("faithfulness", "answer_relevancy",
              "context_precision", "context_recall"):
        v = results["vector"][m].mean() if "vector" in results else float("nan")
        h = results["hybrid"][m].mean() if "hybrid" in results else float("nan")
        print(f"{m:<20}{v:>10.3f}{h:>12.3f}")
    print("=" * 62)
    print("解读：faithfulness 最重要（反幻觉）；0.0~1.0，越高越好")


if __name__ == "__main__":
    main()
