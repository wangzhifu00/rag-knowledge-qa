"""从 PostgreSQL 的 chunk 原文重建 Qdrant 向量库。

为什么单独有这个脚本（这是"原文入 PG"最直接的回报）：
  换 embedding 模型时，旧向量全部作废——新旧模型的向量不在同一个空间，
  混着存等于乱码。但没有这个脚本就得重新上传 PDF、重新解析、重新切分，
  而一旦切分参数被改动，历史结果就不可比了。
  有它之后：改模型 → 跑这个脚本 → 完事。原文一个字不动，切分结果也不动，
  变的只有向量，所以"换模型前后的效果差异"是干净的对比。

做的事：读 PG 全量 chunk → 重新向量化 → 按新维度重建集合 → 灌进去

用法（项目根目录）：
    python scripts/reindex.py                    # 用当前配置的模型重建
    python scripts/reindex.py --model BAAI/bge-m3   # 换个模型重建（做对比实验）
    python scripts/reindex.py --dry-run          # 只报告多少块 / 多少维，不动数据
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.chunks import load_corpus
from rag.retriever import ensure_collection, get_client, upsert_chunks


def main() -> None:
    ap = argparse.ArgumentParser(description="从 PG 的 chunk 原文重建向量库")
    ap.add_argument("--model", default=None,
                    help="指定 embedding 模型（默认用 rag/embedder.py 里的 MODEL_NAME）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只报告规模，不重建")
    args = ap.parse_args()

    rows, total = load_corpus()
    if not rows:
        raise SystemExit("chunks 表是空的——先跑 python scripts/ingest.py 入库")

    # 延迟导入：只有真要重建时才加载模型
    if args.model:
        from rag.embedder import Embedder
        embedder = Embedder(model_name=args.model)
    else:
        from rag.embedder import get_embedder
        embedder = get_embedder()

    print(f"语料：{total} 块（来自 {len({r.filename for r in rows})} 份文档）")
    print(f"模型：{embedder.model_name}（{embedder.dim} 维）")

    if args.dry_run:
        print("[DRY-RUN] 未做任何修改")
        return

    t0 = time.time()
    print("向量化中（CPU 上几百块约十几秒，大模型更久）...")
    vectors = embedder.embed_documents([r.text for r in rows])

    client = get_client()
    # force_recreate：维度可能变了（bge-small 512 → bge-m3 1024），
    # 集合必须按新维度重建，不能往里塞不同维度的向量
    ensure_collection(client, embedder.dim, force_recreate=True)
    count = upsert_chunks(client, [r.id for r in rows], vectors)

    dt = time.time() - t0
    print(f"完成：{count} 个点已写入，耗时 {dt:.1f}s"
          f"（{dt / max(count, 1) * 1000:.0f}ms/块）")
    print("提示：正在跑的服务需要重启一次（BM25 缓存和 collection 维度都是进程内状态）")


if __name__ == "__main__":
    main()
