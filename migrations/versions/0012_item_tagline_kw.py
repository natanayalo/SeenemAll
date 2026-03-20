"""Add tagline and TMDB keywords to items.

Revision ID: 0012_item_tagline_kw
Revises: 0011_catalog_metadata
Create Date: 2026-03-19 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0012_item_tagline_kw"
down_revision = "0011_catalog_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("items", sa.Column("tagline", sa.Text(), nullable=True))
    op.add_column("items", sa.Column("tmdb_keywords", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("items", "tmdb_keywords")
    op.drop_column("items", "tagline")
