"""Restore missing catalog metadata revision placeholder.

Revision ID: 0011_catalog_metadata
Revises: 0008_add_item_maturity_rating
Create Date: 2026-03-19 00:00:00.000000
"""

# revision identifiers, used by Alembic.
revision = "0011_catalog_metadata"
down_revision = "0008_add_item_maturity_rating"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The local development database is already stamped at this revision.
    # Keep this placeholder no-op so Alembic can resolve history cleanly.
    pass


def downgrade() -> None:
    pass
