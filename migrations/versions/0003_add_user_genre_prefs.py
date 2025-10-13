"""add genre_prefs to users

Revision ID: 0003_add_user_genre_prefs
Revises: 0002_add_release_year
Create Date: 2025-10-15

"""

from alembic import op
import sqlalchemy as sa


revision = "0003_add_user_genre_prefs"
down_revision = "0002_add_release_year"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("users", sa.Column("genre_prefs", sa.JSON(), nullable=True))
    op.execute("UPDATE users SET genre_prefs = '{}' WHERE genre_prefs IS NULL")


def downgrade():
    op.drop_column("users", "genre_prefs")
