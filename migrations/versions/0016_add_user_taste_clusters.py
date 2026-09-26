"""Add taste_clusters to users

Revision ID: 0016_add_user_taste_clusters
Revises: 0015_fix_items_tmdb_unique
Create Date: 2026-09-27
"""

from alembic import op
import sqlalchemy as sa


revision = "0016_add_user_taste_clusters"
down_revision = "0015_fix_items_tmdb_unique"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("taste_clusters", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "taste_clusters")
