"""向量化：文本 → 向量（一串数字）。

为什么向量化：计算机没法直接比"哪段话意思接近"，但能比数字。
意思相近的文本，向量在空间里就挨得近——这是检索的数学基础。

当前用 fastembed + BAAI/bge-small-zh-v1.5（本地 CPU，ONNX，无 torch）：
免费、快、文档不出门。路线定的 BGE-M3 等链路通了再换，做对比实验。
"""

from functools import lru_cache

from fastembed import TextEmbedding

# 模型名集中放这，以后换 BGE-M3 只改这一处
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


class Embedder:
    """封装 fastembed。首次使用会下载模型（~100MB，一次性，之后走本地缓存）。"""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        # 用一条样本探测维度，不同模型维度不同（bge-small-zh-v1.5=512）
        sample = next(self._model.embed(["维度探测"]))
        self.dim = len(sample)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化（入库用）。fastembed 内部有批处理，一次传完最快。"""
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """单条向量化（提问用）。"""
        return next(self._model.embed([text])).tolist()


@lru_cache
def get_embedder() -> Embedder:
    """全局单例：模型加载要几秒，且只需加载一次。"""
    return Embedder()
