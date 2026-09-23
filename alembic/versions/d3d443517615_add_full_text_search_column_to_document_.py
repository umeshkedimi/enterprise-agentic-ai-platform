"""Add full-text search column to document chunks

Revision ID: d3d443517615
Revises: 0c8ae1d34f18
Create Date: 2026-09-23 16:12:12.818522

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd3d443517615'
down_revision: Union[str, Sequence[str], None] = '0c8ae1d34f18'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # A GENERATED column, not a plain one written by the application — same
    # posture as the embedding column, whose value the app also never
    # composes by hand. Postgres backfills every existing row's value as
    # part of this ALTER and keeps it in sync on every future INSERT/UPDATE
    # of `content`, so there is no ingestion-side code path that could let
    # this column drift from what it is supposed to index.
    op.execute(
        "ALTER TABLE document_chunks "
        "ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    )
    # `_fts` suffix, not `_hnsw` — both are raw-SQL indexes with no
    # SQLModel/SQLAlchemy equivalent, and both need the `include_object`
    # guard in alembic/env.py to survive a future autogenerate.
    op.execute(
        "CREATE INDEX ix_document_chunks_content_tsv_fts "
        "ON document_chunks USING gin (content_tsv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_content_tsv_fts")
    op.execute("ALTER TABLE document_chunks DROP COLUMN content_tsv")
