from __future__ import annotations

import argparse
import sys

from elasticsearch.exceptions import TransportError

from api.core.elasticsearch_client import create_elasticsearch_client
from etl.elasticsearch_index import ITEMS_INDEX_NAME, ensure_items_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize Elasticsearch indices required for Seen'emAll."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and recreate the items index if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    client = create_elasticsearch_client()
    try:
        ensure_items_index(client, force=args.force)
    except TransportError as exc:
        print(f"Failed to ensure '{ITEMS_INDEX_NAME}' index: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()

    print(f"Elasticsearch index '{ITEMS_INDEX_NAME}' is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
