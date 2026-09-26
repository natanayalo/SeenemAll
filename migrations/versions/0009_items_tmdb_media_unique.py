"""Allow duplicate tmdb IDs across media types

Revision ID: 0009_items_tmdb_media_unique
Revises: 0008_add_item_maturity_rating
Create Date: 2024-11-02 00:00:00.000000
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0009_items_tmdb_media_unique"
down_revision = "0008_add_item_maturity_rating"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_items_tmdb_id", table_name="items")
    op.create_unique_constraint(
        "uq_items_tmdb_media_type", "items", ["tmdb_id", "media_type"]
    )
    op.create_index("ix_items_tmdb_id", "items", ["tmdb_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_items_tmdb_id", table_name="items")
    op.drop_constraint("uq_items_tmdb_media_type", "items", type_="unique")
    op.create_index("ix_items_tmdb_id", "items", ["tmdb_id"], unique=True)
