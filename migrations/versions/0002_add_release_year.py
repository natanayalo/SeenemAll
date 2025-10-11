"""add release_year to items

Revision ID: 0002_add_release_year
Revises: 0001_init
Create Date: 2025-10-14

"""

from alembic import op
import sqlalchemy as sa


revision = "0002_add_release_year"
down_revision = "0001_init"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("items", sa.Column("release_year", sa.Integer(), nullable=True))
    op.create_index("ix_items_release_year", "items", ["release_year"])


def downgrade():
    op.drop_index("ix_items_release_year", table_name="items")
    op.drop_column("items", "release_year")
