"""ORM 表模型：一个类 = 一张表。

Document 表 = RAG 链路的"档案登记处"：
上传一个 PDF，先在这里登记一条记录（谁、多大、解析到哪一步），
然后才是切分(chunker) → 向量化(embedder) → 入 Qdrant。
PG 管元数据和状态，Qdrant 只管向量。

Chunk 表 = 原文的 source of truth（2026-10-26 从 Qdrant payload 挪过来）：
之前原文只存在 Qdrant 的 payload 里，带来三个问题——
  1. 改一个字要整点重写，payload 不是数据库
  2. 想按文档/页统计得全量 scroll（BM25 建索引时就是这么干的）
  3. 换 embedding 模型要重建向量库，原文跟着一起没了，只能重新解析 PDF
挪到 PG 后：Qdrant 只存 (id, 向量)，payload 只剩一个 chunk_id；
原文、页码、所属文档都在 PG 里，能 join、能校验、能随时重建索引。
"""

from datetime import datetime
from sqlalchemy import (BigInteger, DateTime, ForeignKey, Index, Integer,
                        String, Text, UniqueConstraint, func)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#定义模型类   基类+表
class Base(DeclarativeBase):
    """所有模型的公共基类。Base.metadata 会收集全部表结构，Alembic 靠它做迁移。"""
class Document(Base):
    __tablename__ = "documents"
    # 主键，自增。用 BigInteger 是因为 Postgres 的 serial 到 21 亿就爆了
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # 原始文件名（用户上传时叫什么）
    filename: Mapped[str] = mapped_column(String(255), unique=True)

    # 磁盘上的存储路径（data/uploads/xxx.pdf）
    file_path: Mapped[str] = mapped_column(String(512))

    # 文件大小（字节）。BigInteger 防大文件溢出
    file_size: Mapped[int] = mapped_column(BigInteger, default=0)

    # MIME 类型，如 application/pdf
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # 处理状态机：uploaded → parsing → indexed / failed
    status: Mapped[str] = mapped_column(String(20), default="uploaded", index=True)

    # 失败原因（status=failed 时有值）
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 时间戳交给数据库生成（server_default），不用依赖应用时钟
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class Chunk(Base):
    """文档切出来的一块原文。Qdrant 里那个向量的"真身"就是这一行。

    id 直接当 Qdrant 的 point id 用（Qdrant 只接受整数/UUID，
    而 PG 的 bigserial 正好是整数）——这样向量和原文靠同一个编号对上，
    不需要在 payload 里塞文件名+序号再反查。
    """
    __tablename__ = "chunks"
    __table_args__ = (
        # 同一文档内序号唯一：重新入库时按 (document_id, chunk_index) 可以安全替换
        UniqueConstraint("document_id", "chunk_index",
                         name="uq_chunks_doc_index"),
        # 按文档删块是最常见的操作（重新入库/删文档），走这个索引
        Index("ix_chunks_document_id", "document_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True,
                                    autoincrement=True)

    # 所属文档。ondelete="CASCADE"：删文档时块自动跟着删，
    # 不靠应用层记得手动清理（漏一次就留一堆孤儿行）
    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("documents.id", ondelete="CASCADE")
    )

    # 在文件内的顺序号（0 开始）。页码可能跨页重复，序号才是文件内唯一坐标
    chunk_index: Mapped[int] = mapped_column(Integer)

    # 原文。这是整张表存在的理由
    text: Mapped[str] = mapped_column(Text)

    # 涉及的页码（引用溯源用），PG 原生数组，省一张关联表
    page_numbers: Mapped[list[int]] = mapped_column(ARRAY(Integer), default=list)

    # 字符数：调切分参数时用来对比块大小分布，不用每次 length(text)
    char_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
