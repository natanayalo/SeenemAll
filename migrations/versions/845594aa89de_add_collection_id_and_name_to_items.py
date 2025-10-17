"""Add collection metadata columns to items.

Revision ID: 845594aa89de
Revises: 0006_add_item_embedding_version
Create Date: 2025-10-17 10:47:16.045843
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "845594aa89de"
down_revision = "0006_add_item_embedding_version"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("items", sa.Column("collection_id", sa.Integer(), nullable=True))
    op.add_column(
        "items", sa.Column("collection_name", sa.String(length=512), nullable=True)
    )
    op.create_index("ix_items_collection_id", "items", ["collection_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_items_collection_id", table_name="items")
    op.drop_column("items", "collection_name")
    op.drop_column("items", "collection_id")
