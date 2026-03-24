"""Alter embeddings from 384 to 768 dimensions for all-mpnet-base-v2 model.

Revision ID: 0014_alter_embeddings_to_768
Revises: 0013_add_pgvector_index
Create Date: 2026-03-24 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision = "0014_alter_embeddings_to_768"
down_revision = "0013_add_pgvector_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Migrate embeddings from 384 to 768 dimensions.

    This supports switching from all-MiniLM-L6-v2 (384 dims) to
    all-mpnet-base-v2 (768 dims) for better semantic representation.

    Since pgvector Vector type is immutable, we must:
    1. Drop the IVFFLAT index
    2. Drop the old Vector(384) column
    3. Create new Vector(768) column
    4. Re-embedding will populate via ETL
    """
    # Drop IVFFLAT index on item_embeddings first
    op.execute("DROP INDEX IF EXISTS idx_item_embeddings_vector_ivfflat;")

    # Drop vector column from item_embeddings and recreate with new dimension
    op.drop_column("item_embeddings", "vector")
    op.add_column(
        "item_embeddings",
        sa.Column("vector", Vector(768), nullable=False),
    )

    # Drop vector columns from users table and recreate with new dimensions
    op.drop_column("users", "long_vec")
    op.add_column(
        "users",
        sa.Column("long_vec", Vector(768), nullable=True),
    )

    op.drop_column("users", "short_vec")
    op.add_column(
        "users",
        sa.Column("short_vec", Vector(768), nullable=True),
    )


def downgrade() -> None:
    """Revert to 384-dimensional embeddings.

    Reverts schema back to all-MiniLM-L6-v2 (384 dims).
    """
    # Drop 768-dim columns and recreate with 384 dims
    op.drop_column("item_embeddings", "vector")
    op.add_column(
        "item_embeddings",
        sa.Column("vector", Vector(384), nullable=False),
    )

    op.drop_column("users", "long_vec")
    op.add_column(
        "users",
        sa.Column("long_vec", Vector(384), nullable=True),
    )

    op.drop_column("users", "short_vec")
    op.add_column(
        "users",
        sa.Column("short_vec", Vector(384), nullable=True),
    )

    # Recreate IVFFLAT index for downgraded version
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_item_embeddings_vector_ivfflat
        ON item_embeddings
        USING ivfflat (vector vector_cosine_ops)
        WITH (lists = 100);
        """
    )
