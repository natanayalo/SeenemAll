"""add version column to item_embeddings

Revision ID: 0006_add_item_embedding_version
Revises: 0005_add_item_popularity
Create Date: 2025-10-14

"""

from alembic import op
import sqlalchemy as sa


revision = "0006_add_item_embedding_version"
down_revision = "0005_add_item_popularity"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "item_embeddings",
        sa.Column("version", sa.String(length=32), nullable=True),
    )
    op.execute("UPDATE item_embeddings SET version = 'v1' WHERE version IS NULL")
    op.alter_column(
        "item_embeddings",
        "version",
        existing_type=sa.String(length=32),
        nullable=False,
        server_default="v1",
    )
    op.drop_constraint("item_embeddings_item_id_key", "item_embeddings", type_="unique")
    op.create_unique_constraint(
        "uq_item_embedding_item_version",
        "item_embeddings",
        ["item_id", "version"],
    )


def downgrade():
    op.drop_constraint(
        "uq_item_embedding_item_version",
        "item_embeddings",
        type_="unique",
    )
    op.create_unique_constraint(
        "item_embeddings_item_id_key",
        "item_embeddings",
        ["item_id"],
    )
    op.drop_column("item_embeddings", "version")
