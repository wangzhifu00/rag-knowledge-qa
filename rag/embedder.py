"""向量化：文本 → 向量（一串数字）。

为什么向量化：计算机没法直接比"哪段话意思接近"，但能比数字。
意思相近的文本，向量在空间里就挨得近——这是检索的数学基础。

两条后端，按模型名自动选：
  fastembed（默认）   BAAI/bge-small-zh-v1.5，ONNX 推理，不依赖 torch，
                      512 维，CPU 上几十毫秒/块。代价：中文模型，跨语言弱。
  sentence-transformers  BAAI/bge-m3，1024 维，多语言（中/英/日…同一空间），
                      模型 2.3GB，跑起来慢，但中文提问 × 英文语料这种场景
                      是它的主场（bge-small 实测做不到：中文"智能体"和
                      英文 "agent" 在它的空间里离得远）。

为什么要有两条而不是统一一条：fastembed 的 ONNX 导出清单里没有 BGE-M3
（它是 568M 参数的多语言模型，ONNX 化收益不明显）。反过来只用 torch 会让
开发机每次冷启动多等十几秒，而大部分调试根本用不上大模型。

本地目录优先：models/bge-m3/ 存在就直接加载（ModelScope 下载的），
不存在才按模型名在线拉——HF 的 CDN 在国内经常连不上（踩过）。
"""

from functools import lru_cache
from pathlib import Path

from app.config import get_settings

# 本地模型根目录（ModelScope 下载的位置，和 reranker 保持一致）
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# 换模型改 app/config.py 的 embedding_model_name（或 reindex --model 临时覆盖）
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

# 这些模型 fastembed 没有 ONNX 版，必须走 sentence-transformers
_ST_ONLY_KEYWORDS = ("bge-m3", "bge-large", "multilingual-e5", "gte-large")


def _guess_backend(model_name: str) -> str:
    """按模型名猜后端（bge-m3 → st，其余 → fastembed）。"""
    return "st" if any(k in model_name.lower() for k in _ST_ONLY_KEYWORDS) else "fastembed"


def _local_dir(model_name: str) -> Path | None:
    """本地有没有这份模型（models/<短名>/）。有就优先用，省得联网。"""
    path = MODELS_DIR / model_name.split("/")[-1]
    return path if path.exists() else None


class Embedder:
    """封装两种后端，对外只暴露 embed_documents / embed_query / dim / model_name。"""

    def __init__(self, model_name: str | None = None, backend: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        self.backend = backend or _guess_backend(self.model_name)
        self.local_dir = _local_dir(self.model_name)
        # 默认值先给上，st 后端会覆盖（fastembed 走 ONNX，自己管设备）
        self.device = "cpu"
        self.fp16 = False

        if self.backend == "st":
            self._init_st()
        else:
            self._init_fastembed()

        # 用一条样本探测维度，不同模型维度不同（bge-small=512，bge-m3=1024）
        self.dim = len(self.embed_query("维度探测"))

    # ---------- fastembed 后端 ----------
    def _init_fastembed(self) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=self.model_name)

    def _embed_fastembed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    # ---------- sentence-transformers 后端 ----------
    def _init_st(self) -> None:
        from sentence_transformers import SentenceTransformer

        from rag.device import model_kwargs, resolve_device

        settings = get_settings()
        source = str(self.local_dir) if self.local_dir else self.model_name
        self.device = resolve_device(settings.embedding_device)
        self.fp16 = self.device == "cuda" and settings.gpu_fp16
        # 半精度在 from_pretrained 阶段就定下来，避免"fp32 上卡再转 half"
        # 多要出近 1GB 临时显存（4GB 卡上必炸的那一步）
        self._model = SentenceTransformer(
            source,
            device=self.device,
            model_kwargs=model_kwargs(self.device, settings.gpu_fp16),
        )

    def _embed_st(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings：Qdrant 建集合用的是 COSINE，
        # 归一化后内积就等于余弦相似度，分数口径统一在 0~1
        vectors = self._model.encode(
            texts,
            batch_size=8,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 64,   # 大批量给个进度条，小批量不吵
        )
        # fp16 模型输出的是 float16，统一转回 float32 再落库：
        # 上游（Qdrant / numpy / 后续计算）全都按 float32 设计，
        # 混着 fp16 会在这里悄无声息地出精度问题
        return vectors.astype("float32").tolist()

    # ---------- 对外接口 ----------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化（入库用）。"""
        if not texts:
            return []
        if self.backend == "st":
            return self._embed_st(texts)
        return self._embed_fastembed(texts)

    def embed_query(self, text: str) -> list[float]:
        """单条向量化（提问用）。"""
        return self.embed_documents([text])[0]


@lru_cache
def get_embedder() -> Embedder:
    """全局单例：模型加载要几秒，且只需加载一次。"""
    return Embedder()
