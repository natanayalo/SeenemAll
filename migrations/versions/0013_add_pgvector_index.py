"""Add pgvector IVFFLAT index for ANN search performance.

Revision ID: 0013_add_pgvector_index
Revises: 0011_catalog_metadata
"""

from alembic import op


revision = "0013_add_pgvector_index"
down_revision = "0011_catalog_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_item_embeddings_vector_ivfflat
        ON item_embeddings
        USING ivfflat (vector vector_cosine_ops)
        WITH (lists = 100);
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_item_embeddings_vector_ivfflat;")
