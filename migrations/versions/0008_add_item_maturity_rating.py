"""Add maturity rating column to items.

Revision ID: 0008_add_item_maturity_rating
Revises: 845594aa89de
Create Date: 2025-10-20 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0008_add_item_maturity_rating"
down_revision = "845594aa89de"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "items", sa.Column("maturity_rating", sa.String(length=32), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("items", "maturity_rating")
