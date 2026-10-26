"""create chunks table

把 chunk 原文从 Qdrant payload 挪到 PostgreSQL（source of truth）。

为什么这张表值得单独迁移一次：
  Qdrant 的 payload 不是数据库——改一个字要整点重写，想按文档/页统计要全量
  scroll，换 embedding 模型重建集合时原文跟着一起消失（只能重新解析 PDF）。
  原文落 PG 之后，向量变成"可以从原文重算出来的派生数据"，
  scripts/reindex.py 才有意义。

id 用 bigserial 且直接当 Qdrant 的 point id，所以不建额外的外部 id 列。

Revision ID: 8f2a1c9d4e70
Revises: 47cf75b5c164
Create Date: 2026-10-26 15:20:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '8f2a1c9d4e70'
down_revision: Union[str, None] = '47cf75b5c164'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'chunks',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('document_id', sa.BigInteger(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        # PG 原生数组：页码是小整数列表，单开一张关联表不值当
        sa.Column('page_numbers', postgresql.ARRAY(sa.Integer()),
                  nullable=False),
        sa.Column('char_count', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=False),
        # 删文档时块自动跟着删，不靠应用层记得手动清理（漏一次就留一堆孤儿行）
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'],
                                ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('document_id', 'chunk_index',
                            name='uq_chunks_doc_index'),
    )
    op.create_index('ix_chunks_document_id', 'chunks', ['document_id'])


def downgrade() -> None:
    op.drop_index('ix_chunks_document_id', table_name='chunks')
    op.drop_table('chunks')
