"""检索层：Qdrant 的建集合、插入、搜索。

Qdrant 的概念对齐：
  collection ≈ 一张表（本项目的 "documents" 集合）
  point      ≈ 一行（向量 + payload 原文和元数据）
  检索        ≈ "给我离这个问题向量最近的 top_k 个 point"

为什么用 Qdrant 而不是普通数据库：几百上千个向量里找最近邻，
Qdrant 的 HNSW 索引是毫秒级，普通库是全表扫。
"""

from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient, models

from app.config import get_settings
from rag.chunker import Chunk
from rag.embedder import Embedder

settings = get_settings()

# 集合名集中放这（想换名字只改这里）
COLLECTION_NAME = "documents_chunks"


@dataclass
class SearchHit:
    """一条检索结果：原文 + 它是谁 + 相关性得分。"""
    text: str
    filename: str
    page_numbers: list[int]
    score: float
    # chunk 在文件内的序号，配合 filename 可唯一定位一个 chunk（当身份用）。
    # 有些文档代码块跨页重复、前几十字一模一样，所以不能靠原文片段识别。
    chunk_index: int = -1
    # 向量路的原始相似度（RRF 融合后的 score 是名次分，不能当相似度用）。
    # 拒答门槛判断的就是这个字段。纯向量检索时它 == score。
    vector_score: float | None = None


@lru_cache
def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(client: QdrantClient, dim: int, force_recreate: bool = False) -> None:
    """确保集合存在（维度必须和 embedding 模型一致，对不上检索就废了）。

    force_recreate=True 会清空重建——换 embedding 模型后必须重建，
    因为新旧模型的向量不在同一个空间里，混着存等于乱码。
    """
    if force_recreate:
        client.delete_collection(COLLECTION_NAME)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,  # 余弦相似度：文本检索的标准选择
            ),
        )


def delete_points_of_file(client: QdrantClient, filename: str) -> None:
    """按文件名删旧向量——重复入库时先清旧的，防止同一文档存两份。"""
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(
                    key="filename",
                    match=models.MatchValue(value=filename),
                )]
            )
        ),
    )


def upsert_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    vectors: list[list[float]],
    filename: str,
) -> int:
    """向量 + 原文 + 元数据一起写入。payload 存全信息，检索结果才能带引用。"""
    points = [
        models.PointStruct(
            # Qdrant 的 id 只接受整数或 UUID。同一文件入库前会先删旧点，
            # 所以用 chunk_index 当整数 id 不会撞
            id=chunk.chunk_index,
            vector=vector,
            payload={
                "text": chunk.text,
                "filename": filename,
                "page_numbers": chunk.page_numbers,
                "chunk_index": chunk.chunk_index,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def search(
    client: QdrantClient,
    embedder: Embedder,
    query: str,
    top_k: int = 5,
) -> list[SearchHit]:
    """提问 → 问题向量 → 找最近的 top_k 块。"""
    query_vector = embedder.embed_query(query)
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        SearchHit(
            text=p.payload["text"],
            filename=p.payload["filename"],
            page_numbers=p.payload.get("page_numbers", []),
            score=p.score,
            chunk_index=p.payload.get("chunk_index", -1),
            vector_score=p.score,
        )
        for p in hits
    ]


def best_vector_score(hits: list[SearchHit]) -> float:
    """这批结果里最高的向量原始相似度（拒答门槛就用它判断）。

    混合检索下部分 hit 只来自 BM25 路、没有向量分（None），跳过；
    全是 None 说明向量路空手而归，返回 0——必拒。
    """
    values = [h.vector_score for h in hits if h.vector_score is not None]
    return max(values) if values else 0.0
