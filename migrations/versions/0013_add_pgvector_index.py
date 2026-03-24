"""Add pgvector IVFFLAT index for ANN search performance.

Revision ID: 0013_add_pgvector_index
Revises: 0012_item_tagline_kw
Create Date: 2026-03-23 00:00:00.000000
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "0013_add_pgvector_index"
down_revision = "0012_item_tagline_kw"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create IVFFLAT index on item_embeddings.vector for ANN search
    # IVFFLAT is faster to build than HNSW and good for cosine similarity
    # probes=10 is a tuned balance between speed and accuracy
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_item_embeddings_vector_ivfflat
        ON item_embeddings
        USING ivfflat (vector vector_cosine_ops)
        WITH (lists = 100);
        """
    )

    # Also create a HNSW index comment for fallback (can be enabled manually)
    # COMMENT ON INDEX idx_item_embeddings_vector_ivfflat IS 'IVFFLAT index for fast cosine similarity ANN search. Approximate. For exact search, use seq scan.';


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_item_embeddings_vector_ivfflat;")
