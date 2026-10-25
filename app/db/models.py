"""ORM 表模型：一个类 = 一张表。
Document 表 = RAG 链路的"档案登记处"：
上传一个 PDF，先在这里登记一条记录（谁、多大、解析到哪一步），
然后才是切分(chunker) → 向量化(embedder) → 入 Qdrant。
PG 管元数据和状态，Qdrant 只管向量。
"""

from datetime import datetime
from sqlalchemy import BigInteger, DateTime, String, Text, func
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
