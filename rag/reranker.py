"""精排层：交叉编码器(Cross-Encoder)对候选重排。

双塔 vs 交叉编码器（本层存在的理由）：
  向量检索是"双塔"——问题和文档各自编码成向量再比距离，
  快（文档向量可离线算好）但精度有限：实测正负例余弦分重叠
  （0.42~0.55 区间分不开），单一阈值拒答因此失效。
  交叉编码器把「问题+文档」拼在一起喂进模型，词与词的交互
  全程可见，输出的相关度分数区分度远高于双塔。
  代价是慢（每个候选都要现算），所以只用于精排、不做海选——
  工业标准架构：快的海选(向量+BM25 取 top20) → 准的精排(本层挑 top5)。

模型：BAAI/bge-reranker-v2-m3（约2.2GB，多语言，CPU 可跑）
  ⚠️ 不要用 bge-reranker-base：它只支持中-中/英-英同语言对，
  实测"中文提问×英文段落"时打分全是 0.500（sigmoid 零点=没有判断力），
  排序效果等于随机——踩过这个坑（坑13）。v2-m3 是多语言版，
  原生支持跨语言检索场景。
  从 ModelScope 下载到 models/bge-reranker-v2-m3/（国内直连，
  避免重蹈 HF/Xet CDN 被墙的坑8）；目录不存在时才按模型名在线加载。

分数口径：predict 输出原始 logit（约 -10~+10），过 sigmoid 变成
0~1 的相关度概率——拒答门槛判断的就是这个概率，直观且可标定。
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from rag.retriever import SearchHit

# 模型名/目录集中放这，换模型只改这两行
MODEL_DIR_NAME = "bge-reranker-v2-m3"
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
# 本地模型目录（ModelScope 下载解压后的位置）
LOCAL_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / MODEL_DIR_NAME


class Reranker:
    """CrossEncoder 封装。模型加载要十几秒，全局只建一个实例。"""

    def __init__(self):
        # 延迟导入：torch + sentence-transformers 很重，不用 rerank 就不加载
        from sentence_transformers import CrossEncoder

        from app.config import get_settings
        from rag.device import model_kwargs, resolve_device

        settings = get_settings()
        model_path = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else MODEL_NAME
        self.device = resolve_device(settings.rerank_device)
        # 关键：半精度必须在 from_pretrained 阶段生效。先按 fp32 建好再 .to(cuda)
        # 会在搬权重时额外要一份临时显存（实测 978MB），4GB 卡就是这么炸的。
        # 和 BGE-M3 同开时两个模型各 ~1.15GB，合计 2.3GB 才塞得进 4GB。
        self._model = CrossEncoder(
            model_path,
            device=self.device,
            model_kwargs=model_kwargs(self.device, settings.gpu_fp16),
        )
        self.model_path = model_path

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        """对候选重排：返回按 rerank 相关度降序的前 top_k。

        SearchHit.score 被替换为 rerank 相关度概率(0~1)；
        原始向量分保留在 vector_score 里（调试/兜底用）。
        """
        if not hits:
            return []
        pairs = [(query, h.text) for h in hits]
        # batch_size=8：默认一次喂 32 对，3050 Ti 只有 4GB 显存容易 OOM；
        # 小批量对 CPU 速度影响也很小
        logits = self._model.predict(pairs, batch_size=8)
        probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)))

        ranked = sorted(zip(hits, probs), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            SearchHit(
                text=h.text,
                filename=h.filename,
                page_numbers=h.page_numbers,
                score=float(p),  # ← rerank 相关度概率，拒答门槛看它
                chunk_index=h.chunk_index,
                vector_score=h.vector_score,
                chunk_id=h.chunk_id,   # 身份要跟着走，否则融合后认不出是同一块
            )
            for h, p in ranked
        ]


@lru_cache
def get_reranker() -> Reranker:
    """全局单例。"""
    return Reranker()
