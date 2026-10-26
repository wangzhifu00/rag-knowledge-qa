"""检索层：Qdrant 的建集合、插入、搜索。

Qdrant 的概念对齐：
  collection ≈ 一张表（本项目的 "documents_chunks" 集合）
  point      ≈ 一行（向量 + point id）
  检索        ≈ "给我离这个问题向量最近的 top_k 个 point"

**Qdrant 里不存原文**（2026-10-26 改）：point id 就是 PostgreSQL chunks.id，
payload 里只放一个 chunk_id。命中之后拿 id 回 PG 取原文（见 app/db/chunks.py）。
这样换 embedding 模型重建向量库时，PG 里的原文不动，随时可以重算重灌。

为什么用 Qdrant 而不是普通数据库：几百上千个向量里找最近邻，
Qdrant 的 HNSW 索引是毫秒级，普通库是全表扫。
"""

from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient, models

from app.config import get_settings
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
    # chunk 在文件内的序号，展示用（"第 12 页 · 第 3 块"）
    chunk_index: int = -1
    # 向量路的原始相似度（RRF 融合后的 score 是名次分，不能当相似度用）。
    # 拒答门槛判断的就是这个字段。纯向量检索时它 == score。
    vector_score: float | None = None
    # chunks 表主键 = Qdrant point id。原文靠它回 PG 取。
    # -1 表示这条来自临时集合（评估脚本自建，原文就在 payload 里）
    chunk_id: int = -1


@lru_cache
def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collection(client: QdrantClient, dim: int, force_recreate: bool = False) -> None:
    """确保集合存在（维度必须和 embedding 模型一致，对不上检索就废了）。

    force_recreate=True 会清空重建——换 embedding 模型后必须重建，
    因为新旧模型的向量不在同一个空间里，混着存等于乱码。
    """
    if force_recreate and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,  # 余弦相似度：文本检索的标准选择
            ),
        )


def _point_id(p) -> int:
    """Qdrant 的 id 可能是 int 也可能是 UUID 字符串，取整数形式做 chunk_id。"""
    try:
        return int(p.id)
    except (TypeError, ValueError):
        return -1


def upsert_chunks(
    client: QdrantClient,
    chunk_ids: list[int],
    vectors: list[list[float]],
) -> int:
    """只写 id + 向量。原文、页码、文件名全在 PG，Qdrant 不重复存一份。

    point id 直接用 chunks.id：向量和原文靠同一个编号对上，
    不用在 payload 里塞文件名+序号再反查。
    """
    if len(chunk_ids) != len(vectors):
        raise ValueError(f"id 数({len(chunk_ids)})和向量数({len(vectors)})不一致")
    points = [
        models.PointStruct(id=cid, vector=vec, payload={"chunk_id": cid})
        for cid, vec in zip(chunk_ids, vectors)
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def delete_points(client: QdrantClient, point_ids: list[int]) -> None:
    """按 point id 批量删（重新入库后用旧 id 清理残留向量）。"""
    if point_ids:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(points=point_ids),
        )


def hydrate(hits: list[SearchHit]) -> list[SearchHit]:
    """把只有 id 的命中补上原文（原文在 PG，不在 Qdrant）。

    临时集合（评估脚本自建的那种）payload 里本来就带 text，直接跳过。
    """
    missing = [h for h in hits if not h.text and h.chunk_id >= 0]
    if not missing:
        return hits

    # 延迟导入：只用临时集合的脚本（如切分参数消融）不必连 PG
    from app.db.chunks import fetch_by_ids

    rows = fetch_by_ids([h.chunk_id for h in missing])
    for hit in missing:
        row = rows.get(hit.chunk_id)
        if row is None:
            # PG 里没有这块 = Qdrant 残留了旧向量（重新入库后没清干净）。
            # 丢掉它，别让空原文流进 prompt 或 rerank。
            continue
        hit.text = row.text
        hit.filename = row.filename
        hit.page_numbers = row.page_numbers
        hit.chunk_index = row.chunk_index
    return [h for h in hits if h.text]


def search(
    client: QdrantClient,
    embedder: Embedder,
    query: str,
    top_k: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> list[SearchHit]:
    """提问 → 问题向量 → 找最近的 top_k 块 → 回 PG 补原文。

    collection_name 默认用正式集合；评估脚本会传临时集合进来对比。
    """
    query_vector = embedder.embed_query(query)
    points = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    hits = []
    for p in points:
        payload = p.payload or {}
        hits.append(SearchHit(
            text=payload.get("text", ""),   # 老结构/临时集合才有；新结构靠 hydrate
            filename=payload.get("filename", ""),
            page_numbers=payload.get("page_numbers", []),
            score=p.score,
            chunk_index=payload.get("chunk_index", -1),
            vector_score=p.score,
            chunk_id=_point_id(p),
        ))
    return hydrate(hits)


def best_vector_score(hits: list[SearchHit]) -> float:
    """这批结果里最高的向量原始相似度（拒答门槛就用它判断）。

    混合检索下部分 hit 只来自 BM25 路、没有向量分（None），跳过；
    全是 None 说明向量路空手而归，返回 0——必拒。
    """
    values = [h.vector_score for h in hits if h.vector_score is not None]
    return max(values) if values else 0.0
