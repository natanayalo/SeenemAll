"""Track catalog changes that must be reflected in Elasticsearch.

Revision ID: 0017_catalog_es_sync_queue
Revises: 0016_add_user_taste_clusters
Create Date: 2026-09-27
"""

from alembic import op
import sqlalchemy as sa


revision = "0017_catalog_es_sync_queue"
down_revision = "0016_add_user_taste_clusters"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "catalog_es_sync_queue",
        sa.Column("item_id", sa.Integer(), primary_key=True),
        sa.Column(
            "changed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("clock_timestamp()"),
        ),
    )
    op.create_index(
        "ix_catalog_es_sync_queue_changed_at",
        "catalog_es_sync_queue",
        ["changed_at"],
    )
    op.execute(
        """
        CREATE FUNCTION enqueue_catalog_es_sync() RETURNS trigger AS $$
        DECLARE
            row_data jsonb;
            changed_item_id integer;
        BEGIN
            row_data := CASE WHEN TG_OP = 'DELETE' THEN to_jsonb(OLD) ELSE to_jsonb(NEW) END;
            IF TG_TABLE_NAME = 'items' THEN
                changed_item_id := (row_data ->> 'id')::integer;
            ELSE
                changed_item_id := (row_data ->> 'item_id')::integer;
            END IF;

            INSERT INTO catalog_es_sync_queue (item_id, changed_at)
            VALUES (changed_item_id, clock_timestamp())
            ON CONFLICT (item_id) DO UPDATE
            SET changed_at = GREATEST(catalog_es_sync_queue.changed_at, EXCLUDED.changed_at);

            IF TG_OP = 'DELETE' THEN
                RETURN OLD;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    for table_name in ("items", "item_embeddings", "availability"):
        op.execute(
            f"""
            CREATE TRIGGER {table_name}_enqueue_es_sync
            AFTER INSERT OR UPDATE OR DELETE ON {table_name}
            FOR EACH ROW EXECUTE FUNCTION enqueue_catalog_es_sync();
            """
        )
    op.execute(
        """
        INSERT INTO catalog_es_sync_queue (item_id)
        SELECT id FROM items
        ON CONFLICT (item_id) DO NOTHING;
        """
    )


def downgrade() -> None:
    for table_name in ("items", "item_embeddings", "availability"):
        op.execute(
            f"DROP TRIGGER IF EXISTS {table_name}_enqueue_es_sync ON {table_name}"
        )
    op.execute("DROP FUNCTION IF EXISTS enqueue_catalog_es_sync()")
    op.drop_index(
        "ix_catalog_es_sync_queue_changed_at", table_name="catalog_es_sync_queue"
    )
    op.drop_table("catalog_es_sync_queue")
