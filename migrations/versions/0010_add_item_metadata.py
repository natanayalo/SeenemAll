"""Add cast/crew/keywords/spoken_languages columns

Revision ID: 0010_add_item_metadata
Revises: 0009_items_tmdb_media_unique
Create Date: 2025-02-21 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0010_add_item_metadata"
down_revision = "0009_items_tmdb_media_unique"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("items", sa.Column("cast", sa.JSON(), nullable=True))
    op.add_column("items", sa.Column("directors", sa.JSON(), nullable=True))
    op.add_column("items", sa.Column("producers", sa.JSON(), nullable=True))
    op.add_column("items", sa.Column("writers", sa.JSON(), nullable=True))
    op.add_column("items", sa.Column("keywords", sa.JSON(), nullable=True))
    op.add_column("items", sa.Column("spoken_languages", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("items", "spoken_languages")
    op.drop_column("items", "keywords")
    op.drop_column("items", "writers")
    op.drop_column("items", "producers")
    op.drop_column("items", "directors")
    op.drop_column("items", "cast")
