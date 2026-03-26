from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from dotenv import load_dotenv
from sqlalchemy import select

from api.db.models import Item
from api.db.session import get_engine

load_dotenv()


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(path: Path, payload: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


def fetch_existing_tmdb_ids(tmdb_ids: Set[int]) -> Set[int]:
    if not tmdb_ids:
        return set()
    engine = get_engine()
    stmt = select(Item.tmdb_id).where(Item.tmdb_id.in_(tmdb_ids))
    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
    return {row[0] for row in rows}


def merge_candidates(evaluation_path: Path, candidates_path: Path) -> Dict[str, Any]:
    evaluation_entries = load_json(evaluation_path)
    candidate_entries = load_json(candidates_path)

    main_index: Dict[str, Dict[str, Any]] = {
        entry["query"]: entry for entry in evaluation_entries
    }

    all_candidate_ids: Set[int] = set()
    for entry in candidate_entries:
        for item in entry.get("golden_set", []):
            tmdb_id = item.get("id")
            if isinstance(tmdb_id, int):
                all_candidate_ids.add(tmdb_id)

    present_ids = fetch_existing_tmdb_ids(all_candidate_ids)

    added_count = 0
    remaining_entries: List[Dict[str, Any]] = []

    for entry in candidate_entries:
        query = entry.get("query")
        if not query:
            continue

        main_entry = main_index.get(query)
        if main_entry is None:
            main_entry = {k: v for k, v in entry.items() if k != "golden_set"}
            main_entry["golden_set"] = []
            evaluation_entries.append(main_entry)
            main_index[query] = main_entry

        existing_ids = {
            item["id"]
            for item in main_entry.get("golden_set", [])
            if isinstance(item, dict) and "id" in item
        }

        carried_over: List[Dict[str, Any]] = []
        for item in entry.get("golden_set", []):
            tmdb_id = item.get("id")
            if not isinstance(tmdb_id, int):
                carried_over.append(item)
                continue
            if tmdb_id in present_ids and tmdb_id not in existing_ids:
                main_entry.setdefault("golden_set", []).append(item)
                existing_ids.add(tmdb_id)
                added_count += 1
            else:
                if tmdb_id not in existing_ids:
                    carried_over.append(item)

        if carried_over:
            new_entry = {k: v for k, v in entry.items()}
            new_entry["golden_set"] = carried_over
            remaining_entries.append(new_entry)

    evaluation_entries.sort(key=lambda entry: entry.get("query", ""))

    save_json(evaluation_path, evaluation_entries)
    save_json(candidates_path, remaining_entries)

    return {
        "added": added_count,
        "remaining": sum(
            len(entry.get("golden_set", [])) for entry in remaining_entries
        ),
        "present_ids": len(present_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge candidate golden-set entries that already exist in the catalog."
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("evaluation/evaluation_set.json"),
        help="Path to the main evaluation set JSON.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Path to candidate golden-set JSON file to process.",
    )
    args = parser.parse_args()

    stats = merge_candidates(args.evaluation, args.candidates)
    print(
        f"Added {stats['added']} titles from {args.candidates}. "
        f"{stats['remaining']} candidates remain for manual follow-up."
    )


if __name__ == "__main__":
    main()
