"""混合检索：BM25（关键词）+ 向量（语义）→ RRF 融合排名。

为什么需要两路检索：
  向量检索懂"意思"——"文档怎么切" 能搜到讲 "chunking 策略" 的段落；
  但它对专有名词不灵——"SelfQueryRetriever" 这种精确术语，
  和提问向量的距离未必近，可能排不到前面。
  BM25 是经典关键词算法（TF-IDF 的升级版）：词对上了才给分，
  恰好补上向量的短板。

为什么用 RRF（Reciprocal Rank Fusion）融合：
  两路的原始分数没法直接相加——BM25 分数可能是 8.7，
  余弦相似度只有 0~1，量纲完全不同。
  RRF 的思路是不看分数、只看名次：
      每条结果的贡献分 = 1 / (k + 名次)，k 常取 60
  两路的名次分各自累加，总分高者排前。
  这是 Elasticsearch / OpenSearch 内置的标准融合方法。

工程细节——BM25 索引放内存：
  chunk 全文存在 Qdrant 的 payload 里，BM25 建索引时全量拉一遍，
  分词后缓存在进程内存。语料只有几百块，秒建；
  用"集合点数"判断语料有没有变（新文档入库），变了自动重建索引。
"""

from functools import lru_cache
import jieba
from rank_bm25 import BM25Okapi
from rag.retriever import COLLECTION_NAME, SearchHit, get_client

# 每路检索先取这么多个候选，融合后再截取 top_k
# （两边都取多一点，融合才有"合并意见"的空间）
CANDIDATE_DEPTH = 20

# RRF 的 k：越大名次差距越平滑，60 是论文推荐值（业界默认）
RRF_K = 60

# BM25 索引缓存：{"point_count": 上次建索引时的语料点数, "bm25": 索引, ...}
_bm25_cache: dict = {}


def _tokenize(text: str) -> list[str]:
    """分词：BM25 按词匹配，中文必须先切词（英文按空格天然切好）。

    jieba 的搜索引擎模式：在普通模式基础上再切出更细的粒度，
    比如 "检索评估" 会额外给出 "检索" "评估"——提高召回。
    中英混排（我们的 PDF 就是）都能处理。
    """
    return [t for t in jieba.cut_for_search(text) if t.strip()]


def _load_corpus(collection_name: str) -> tuple[list[SearchHit], int]:
    """从 Qdrant 全量拉 chunk（原文在 payload 里，它是当前唯一存放地）。

    返回 (所有 chunk 的 SearchHit 列表, 集合总点数)。
    """
    client = get_client()
    points, offset = client.scroll(
        collection_name=collection_name,
        limit=256,
        with_payload=True,
        with_vectors=False,  # 建关键词索引用不上向量，不拉省内存
    )
    while offset is not None:
        more, offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(more)

    total = client.count(collection_name=collection_name, exact=True).count
    hits = [
        SearchHit(
            text=p.payload["text"],
            filename=p.payload.get("filename", collection_name),
            page_numbers=p.payload.get("page_numbers", []),
            score=0.0,  # 此时还没有分数，BM25 检索时才算
            chunk_index=p.payload.get("chunk_index", -1),
        )
        for p in points
    ]
    return hits, total


def _get_bm25(collection_name: str) -> tuple[BM25Okapi | None, list[SearchHit]]:
    """拿 BM25 索引（按集合分别缓存）。语料点数变了自动重建。

    评估脚本会用临时集合跑不同参数，所以缓存必须按集合名区分，
    不能只按点数——不同配置的块数可能碰巧一样。
    """
    corpus, total = _load_corpus(collection_name)
    if not corpus:
        return None, []

    cache = _bm25_cache.setdefault(collection_name, {})
    if cache.get("point_count") == total:
        return cache["bm25"], cache["corpus"]

    tokenized = [_tokenize(hit.text) for hit in corpus]
    bm25 = BM25Okapi(tokenized)
    cache.clear()
    cache.update({"point_count": total, "bm25": bm25, "corpus": corpus})
    return bm25, corpus


def _bm25_search(query: str, top_k: int, collection_name: str = COLLECTION_NAME) -> list[SearchHit]:
    """关键词检索：分词 → BM25 打分 → 取前 top_k。"""
    bm25, corpus = _get_bm25(collection_name)
    if bm25 is None:
        return []
    tokens = _tokenize(query)
    if not tokens:  # 问题切不出任何词（极端情况）
        return []
    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)
    results = []
    for i in ranked[:top_k]:
        if scores[i] <= 0:  # 一个词都对不上，别凑数
            break
        results.append(
            SearchHit(
                text=corpus[i].text,
                filename=corpus[i].filename,
                page_numbers=corpus[i].page_numbers,
                score=float(scores[i]),
                chunk_index=corpus[i].chunk_index,
            )
        )
    return results


def _hit_key(hit: SearchHit) -> tuple:
    """chunk 的身份标识：(文件名, chunk_index)。

    不能拿原文片段当身份——这份 PDF 里有跨页重复的代码块，
    前 80 字符完全相同，会把两个不同的 chunk 当成一个，
    分数重复累加，排名被虚假抬高（踩过这个坑）。
    """
    if hit.chunk_index >= 0:
        return (hit.filename, hit.chunk_index)
    return (hit.filename, hit.text[:80])  # 兜底：老数据没有 chunk_index


def _rrf_fuse(rankings: list[list[SearchHit]], top_k: int) -> list[SearchHit]:
    """RRF 融合：每路结果的 1/(k+名次) 累加。

    同一块同时出现在两路结果里，分数才会合并到一起。
    """
    scores: dict[tuple, float] = {}
    hit_map: dict[tuple, SearchHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):  # 名次从 1 开始
            key = _hit_key(hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            hit_map.setdefault(key, hit)

    fused_keys = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]
    return [
        SearchHit(
            text=hit_map[k].text,
            filename=hit_map[k].filename,
            page_numbers=hit_map[k].page_numbers,
            score=round(scores[k], 4),  # RRF 分数（供展示，别和相似度混淆）
            chunk_index=hit_map[k].chunk_index,
        )
        for k in fused_keys
    ]


def hybrid_search(
    query: str,
    embedder,
    top_k: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> list[SearchHit]:
    """混合检索主入口：向量一路 + BM25 一路 → RRF 融合。

    注意 RRF 后的 score 是融合名次分，不是相似度——
    它只用于排序，数值大小没有"相似程度"的直观含义。
    """
    # 延迟导入避免循环依赖：retriever.search 里用不到 hybrid
    from rag.retriever import search as vector_search

    client = get_client()
    vector_hits = vector_search(client, embedder, query, top_k=CANDIDATE_DEPTH,
                                collection_name=collection_name)
    keyword_hits = _bm25_search(query, top_k=CANDIDATE_DEPTH,
                                collection_name=collection_name)

    fused = _rrf_fuse([vector_hits, keyword_hits], top_k=top_k)

    # 把向量原始分贴回融合结果——拒答门槛判断的是相似度，不是 RRF 名次分
    vector_scores = {_hit_key(h): h.vector_score for h in vector_hits}
    for hit in fused:
        if hit.vector_score is None:
            hit.vector_score = vector_scores.get(_hit_key(hit))

    return fused


@lru_cache
def warmup_jieba() -> None:
    """预热分词器：jieba 首次分词要加载词典（约 1 秒），服务启动时提前做掉。"""
    jieba.initialize()
