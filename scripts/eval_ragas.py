"""RAGAS 答案质量评估：评估的"决赛"。

前面 eval_retrieval.py 量的是"检索找得准不准"（第一层，免费、秒出），
本脚本用 RAGAS 量"最终答案质量好不好"（第二层，裁判是 LLM，烧 API）：

  faithfulness       忠实度：答案是否都有检索材料支撑（反幻觉，最关键）
  answer_relevancy   切题度：答案是否对准了问题
  context_precision  检索精度：该排前的材料排得靠不靠前
  context_recall     检索召回：该有的材料都进上下文了吗

设计：
  - 只评 7 道有答案的题。应拒答题没有参考答案，RAGAS 打不了分，
    它们的表现已由第一层评估的"拒答正确率"覆盖
  - 答案不是人写的，是系统真实生成的（generate_answer），检索也走
    真实链路——评的就是线上行为本身
  - vector / hybrid 两种模式各跑一遍，对比哪个配置答案质量更高
  - 裁判 LLM 和评估用向量化都用 .env 里配的供应商（OpenAI 兼容协议）

用 ragas 0.4 的新式指标 API（ragas.metrics.collections）：
  - 指标是"实例"不是"单例"，裁判 LLM/embedding 在构造时注入
  - 底层用 instructor 结构化输出，不再有 ragas.metrics 旧单例的
    DeprecationWarning，也不依赖 langchain_community（坑12 的根源）
  - 每个指标收的字段不同（见 METRIC_FIELDS），已逐个实测确认

用法（项目根目录）：
    python scripts/eval_ragas.py

成本提示：每个指标每题要调若干次裁判 LLM，全量约一两百次调用，
qwen-plus 计费下大约几毛钱，跑一次几分钟。
"""

import asyncio
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

# 7 道有答案题的参考答案（依据文档原文人工提炼，供 RAGAS 对比打分）
# 注意：原 8 题里的「DeepAgents是什么」已剔除（2026-10-26）——
# 复核语料后确认全文没有一句定义 Deep Agents，只有 URL/代码/导航链接，
# 它属于"只提到名称、没给定义"，正确行为是拒答；
# 原先的参考答案是我按常识补的，拿它当尺子等于惩罚正确行为
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

# 各指标 ascore() 需要的字段（0.4 新 API 逐个实测确认，字段不对直接 TypeError）
METRIC_FIELDS = {
    "faithfulness":      ("user_input", "response", "retrieved_contexts"),
    "answer_relevancy":  ("user_input", "response"),
    "context_precision": ("user_input", "retrieved_contexts", "reference"),
    "context_recall":    ("user_input", "retrieved_contexts", "reference"),
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


async def score_mode(records: list[dict], metrics: dict) -> dict[str, list[float]]:
    """逐题逐指标打分，返回 {指标名: [每题得分]}。

    新式指标要求异步客户端 + await ascore(...)，
    这里串行 await（不并发），控制裁判调用的节奏。

    单条失败不中断整轮：记 nan 继续跑。真实踩过的坑——
    faithfulness 要把答案拆成若干论断逐条判定，长答案的判定 JSON
    超过裁判 max_tokens 会抛 IncompleteOutputException（见坑19）。
    """
    scores = {name: [] for name in metrics}
    for record in records:
        row = {}
        for name, metric in metrics.items():
            kwargs = {f: record[f] for f in METRIC_FIELDS[name]}
            try:
                result = await metric.ascore(**kwargs)
                value = result.value if hasattr(result, "value") else float(result)
            except Exception as exc:  # noqa: BLE001 - 单条失败不该毁掉整轮
                print(f"  [WARN] {name} 在「{record['user_input']}」上失败: "
                      f"{type(exc).__name__}: {exc}")
                value = float("nan")
            scores[name].append(value)
            row[name] = value
        print("  " + record["user_input"] + "  " +
              "  ".join(f"{n}={v:.3f}" for n, v in row.items()))
    return scores


def main() -> None:
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (AnswerRelevancy, ContextPrecision,
                                           ContextRecall, Faithfulness)

    # 裁判 LLM：就是业务用的那家（OpenAI 兼容协议，换 base_url 即可）
    # 注意必须给 AsyncOpenAI——新式指标的 ascore 走异步通道，
    # 传同步客户端会报 "Cannot use agenerate() with a synchronous client"
    aclient = AsyncOpenAI(api_key=settings.llm_api_key,
                          base_url=settings.llm_base_url)
    judge_llm = llm_factory(model=settings.llm_model, provider="openai",
                            client=aclient)
    # max_tokens 必须显式放大：llm_factory 造出的 InstructorLLM 默认只有 1024
    # （InstructorModelArgs 的默认值），而 faithfulness 要把长答案拆成几十条
    # 论断逐条判定，判定结果 JSON 一超限就抛 IncompleteOutputException，
    # 整轮评估直接崩（坑19）。llm_factory 不接受 model_args 参数，只能构造后改；
    # 0.4.3 里 model_args 是 dict（换成 pydantic 对象的写法也一并兼容）。
    if isinstance(judge_llm.model_args, dict):
        judge_llm.model_args["temperature"] = 0
        judge_llm.model_args["max_tokens"] = 4096
    else:
        judge_llm.model_args.temperature = 0
        judge_llm.model_args.max_tokens = 4096
    # 切题度指标需要向量化答案，用同一家 embedding 服务
    judge_embeddings = RagasOpenAIEmbeddings(client=aclient,
                                             model="text-embedding-v3")

    metrics = {
        "faithfulness": Faithfulness(llm=judge_llm),
        "answer_relevancy": AnswerRelevancy(llm=judge_llm,
                                            embeddings=judge_embeddings),
        "context_precision": ContextPrecision(llm=judge_llm),
        "context_recall": ContextRecall(llm=judge_llm),
    }
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
        results[mode] = asyncio.run(score_mode(records, metrics))
        print(f"  ({time.time() - t0:.0f}s)")

    # 汇总对比：两个模式各指标的平均分（跳过失败条目的 nan）
    def mean(values: list[float]) -> float:
        valid = [v for v in values if v == v]   # 过滤 nan
        return sum(valid) / len(valid) if valid else float("nan")

    print("\n" + "=" * 62)
    print(f"{'指标':<20}{'纯向量':>10}{'混合(RRF)':>12}")
    print("-" * 62)
    for m in METRIC_FIELDS:
        v = mean(results["vector"][m]) if "vector" in results else float("nan")
        h = mean(results["hybrid"][m]) if "hybrid" in results else float("nan")
        print(f"{m:<20}{v:>10.3f}{h:>12.3f}")
    print("=" * 62)
    print("解读：faithfulness 最重要（反幻觉）；0.0~1.0，越高越好")


if __name__ == "__main__":
    main()
