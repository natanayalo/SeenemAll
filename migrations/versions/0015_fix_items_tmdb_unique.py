"""Fix items unique constraint to be (tmdb_id, media_type)

Revision ID: 0015_fix_items_tmdb_unique
Revises: 0014_reconcile_catalog_schema
Create Date: 2026-09-26
"""

from alembic import op


revision = "0015_fix_items_tmdb_unique"
down_revision = "0014_reconcile_catalog_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop legacy single-column unique constraint on tmdb_id if present
    op.execute("ALTER TABLE items DROP CONSTRAINT IF EXISTS items_tmdb_id_key")
    # Ensure composite unique constraint on (tmdb_id, media_type)
    op.execute(
        "ALTER TABLE items ADD CONSTRAINT uq_items_tmdb_media_type UNIQUE (tmdb_id, media_type)"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE items DROP CONSTRAINT IF EXISTS uq_items_tmdb_media_type")
    op.execute("ALTER TABLE items ADD CONSTRAINT items_tmdb_id_key UNIQUE (tmdb_id)")
