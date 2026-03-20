from __future__ import annotations

import argparse

from etl.tmdb_sync import run_metadata_backfill


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill TMDB tagline/keyword metadata for existing catalog rows."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of items to backfill.",
    )
    parser.add_argument(
        "--all-items",
        action="store_true",
        help="Backfill all missing-metadata items, not only ranked catalog rows.",
    )
    args = parser.parse_args()
    updated = run_metadata_backfill(
        limit=args.limit,
        ranked_only=not args.all_items,
    )
    print(f"updated_items={updated}")


if __name__ == "__main__":
    main()
