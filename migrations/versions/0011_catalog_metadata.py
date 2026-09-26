"""Add catalog metadata table for dynamic alias lists

Revision ID: 0011_catalog_metadata
Revises: 0010_add_item_metadata
Create Date: 2025-02-24 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import table, column
from sqlalchemy import String, JSON

# revision identifiers, used by Alembic.
revision = "0011_catalog_metadata"
down_revision = "0010_add_item_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "catalog_metadata",
        sa.Column("key", sa.String(length=64), primary_key=True),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )

    metadata_table = table(
        "catalog_metadata",
        column("key", String),
        column("data", JSON),
    )

    op.bulk_insert(
        metadata_table,
        [
            {
                "key": "streaming_provider_aliases",
                "data": {
                    "netflix": ["netflix", "nfx"],
                    "disney_plus": ["disney_plus", "disney", "dnp"],
                    "prime_video": [
                        "prime_video",
                        "primevideo",
                        "amazon",
                        "amz",
                        "amp",
                    ],
                    "hulu": ["hulu", "hlu"],
                    "max": ["max", "hbomax", "hbo", "hbm"],
                    "apple_tv_plus": ["apple_tv_plus", "appletvplus", "apple", "atp"],
                    "paramount_plus": [
                        "paramount_plus",
                        "paramountplus",
                        "prm",
                        "pmnt",
                        "paramount",
                    ],
                },
            },
            {
                "key": "top_query_keywords",
                "data": [
                    "best",
                    "best of",
                    "must watch",
                    "must-watch",
                    "greatest",
                    "top",
                    "classic",
                    "award-winning",
                ],
            },
        ],
    )


def downgrade() -> None:
    op.drop_table("catalog_metadata")
