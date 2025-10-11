"""init schema

Revision ID: 0001_init
Revises:
Create Date: 2025-10-11

"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "items",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tmdb_id", sa.Integer(), nullable=False, unique=True),
        sa.Column("media_type", sa.String(length=10), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("overview", sa.Text(), nullable=True),
        sa.Column("runtime", sa.Integer(), nullable=True),
        sa.Column("original_language", sa.String(length=16), nullable=True),
        sa.Column("genres", sa.JSON(), nullable=True),
        sa.Column("poster_url", sa.String(length=1024), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")
        ),
    )

    op.create_table(
        "item_embeddings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "item_id",
            sa.Integer(),
            sa.ForeignKey("items.id", ondelete="CASCADE"),
            unique=True,
        ),
        sa.Column("vector", Vector(384), nullable=False),
    )

    op.create_table(
        "availability",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "item_id", sa.Integer(), sa.ForeignKey("items.id", ondelete="CASCADE")
        ),
        sa.Column("country", sa.String(length=8)),
        sa.Column("service", sa.String(length=64)),
        sa.Column("offer_type", sa.String(length=16)),
        sa.Column("deeplink", sa.String(length=1024)),
        sa.Column("web_url", sa.String(length=1024)),
        sa.Column(
            "last_checked", sa.DateTime(timezone=True), server_default=sa.text("NOW()")
        ),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.String(length=128), unique=True),
        sa.Column("long_vec", Vector(384), nullable=True),
        sa.Column("short_vec", Vector(384), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")
        ),
    )

    op.create_table(
        "user_history",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.String(length=128)),
        sa.Column("item_id", sa.Integer()),
        sa.Column("event_type", sa.String(length=32)),
        sa.Column("weight", sa.Integer(), server_default="1"),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.String(length=128), nullable=True),
        sa.Column("item_id", sa.Integer(), nullable=True),
        sa.Column("type", sa.String(length=32)),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("meta", sa.JSON(), nullable=True),
    )


def downgrade():
    op.drop_table("feedback")
    op.drop_table("user_history")
    op.drop_table("users")
    op.drop_table("availability")
    op.drop_table("item_embeddings")
    op.drop_table("items")
