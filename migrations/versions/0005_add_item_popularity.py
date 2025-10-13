"""add popularity and rank fields to items

Revision ID: 0005_add_item_popularity
Revises: 0004_add_user_neighbors
Create Date: 2025-10-14

"""

from alembic import op
import sqlalchemy as sa


revision = "0005_add_item_popularity"
down_revision = "0004_add_user_neighbors"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("items", sa.Column("popularity", sa.Float(), nullable=True))
    op.add_column("items", sa.Column("vote_average", sa.Float(), nullable=True))
    op.add_column("items", sa.Column("vote_count", sa.Integer(), nullable=True))
    op.add_column("items", sa.Column("popular_rank", sa.Integer(), nullable=True))
    op.add_column("items", sa.Column("trending_rank", sa.Integer(), nullable=True))
    op.add_column("items", sa.Column("top_rated_rank", sa.Integer(), nullable=True))


def downgrade():
    op.drop_column("items", "top_rated_rank")
    op.drop_column("items", "trending_rank")
    op.drop_column("items", "popular_rank")
    op.drop_column("items", "vote_count")
    op.drop_column("items", "vote_average")
    op.drop_column("items", "popularity")
