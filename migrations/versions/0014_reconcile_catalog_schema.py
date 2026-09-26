"""Reconcile catalog metadata columns for existing databases.

Revision ID: 0014_reconcile_catalog_schema
Revises: 0013_add_pgvector_index

This migration is intentionally compatible with the existing 384-dimensional
embedding schema. It repairs databases that were stamped past metadata
migrations without applying them.
"""

from alembic import op


revision = "0014_reconcile_catalog_schema"
down_revision = "0013_add_pgvector_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute('ALTER TABLE items ADD COLUMN IF NOT EXISTS "cast" JSON')
    op.execute("ALTER TABLE items ADD COLUMN IF NOT EXISTS directors JSON")
    op.execute("ALTER TABLE items ADD COLUMN IF NOT EXISTS producers JSON")
    op.execute("ALTER TABLE items ADD COLUMN IF NOT EXISTS writers JSON")
    op.execute("ALTER TABLE items ADD COLUMN IF NOT EXISTS keywords JSON")
    op.execute("ALTER TABLE items ADD COLUMN IF NOT EXISTS spoken_languages JSON")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS catalog_metadata (
            key VARCHAR(64) PRIMARY KEY,
            data JSON NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS catalog_metadata")
    op.execute("ALTER TABLE items DROP COLUMN IF EXISTS spoken_languages")
    op.execute("ALTER TABLE items DROP COLUMN IF EXISTS keywords")
    op.execute("ALTER TABLE items DROP COLUMN IF EXISTS writers")
    op.execute("ALTER TABLE items DROP COLUMN IF EXISTS producers")
    op.execute("ALTER TABLE items DROP COLUMN IF EXISTS directors")
    op.execute('ALTER TABLE items DROP COLUMN IF EXISTS "cast"')
