"""add neighbors to users

Revision ID: 0004_add_user_neighbors
Revises: 0003_add_user_genre_prefs
Create Date: 2025-10-13

"""

from alembic import op
import sqlalchemy as sa


revision = "0004_add_user_neighbors"
down_revision = "0003_add_user_genre_prefs"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("users", sa.Column("neighbors", sa.JSON(), nullable=True))
    op.execute("UPDATE users SET neighbors = '[]' WHERE neighbors IS NULL")


def downgrade():
    op.drop_column("users", "neighbors")
