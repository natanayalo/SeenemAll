from __future__ import annotations

import argparse

from etl.justwatch_sync import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync streaming availability from JustWatch."
    )
    parser.add_argument(
        "--country",
        help="ISO country code, defaults to JUSTWATCH_COUNTRY env or IL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of catalog items to process.",
    )
    args = parser.parse_args()
    run(country=args.country, limit=args.limit)


if __name__ == "__main__":
    main()
