"""Normalize evaluation-set metadata for stable offline comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

CONSTRAINT_TAGS = {
    "provider_filtered",
    "runtime_capped",
    "maturity_capped",
    "language_or_region",
    "year_or_decade",
}


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalized_tag_list(entry: Dict[str, Any]) -> List[str]:
    raw = entry.get("slice_tags", [])
    if not isinstance(raw, list):
        return []
    cleaned: List[str] = []
    seen: set[str] = set()
    for tag in raw:
        value = str(tag).strip()
        if not value or value in seen:
            continue
        cleaned.append(value)
        seen.add(value)
    return cleaned


def _sync_query_tags(tags: List[str], *, has_query: bool) -> List[str]:
    next_tags = [tag for tag in tags if tag not in {"query", "no_query"}]
    if has_query:
        next_tags.append("query")
    else:
        next_tags.append("no_query")
    return next_tags


def _infer_bucket(entry: Dict[str, Any]) -> str:
    tags = set(_normalized_tag_list(entry))
    query = entry.get("query")
    has_query = query not in (None, "")

    if not has_query:
        if "cold_start" in tags:
            return "no_query_cold_start"
        return "no_query_warm_start"

    if "edge_adversarial" in tags:
        return "edge_adversarial"
    if "profile_persona" in tags:
        return "profile_persona"

    constraint_count = sum(1 for tag in CONSTRAINT_TAGS if tag in tags)
    if constraint_count >= 2:
        return "constrained_multi_filter"
    if constraint_count == 1:
        return "constrained_single_filter"

    has_movie = "movie" in tags
    has_tv = "tv" in tags
    if has_movie and not has_tv:
        return "semantic_query_movie"
    if has_tv and not has_movie:
        return "semantic_query_tv"
    if has_movie and has_tv:
        return "constrained_multi_filter"
    return "semantic_query_movie"


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize evaluation-set metadata.")
    parser.add_argument(
        "--input",
        default="evaluation/evaluation_set_v2.json",
        help="Input evaluation set JSON.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON path (defaults to in-place update).",
    )
    parser.add_argument(
        "--case-prefix",
        default="v2_case",
        help="Prefix used for generated case_ids.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based starting index for generated case_ids.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write file; print only summary.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    payload = _load_json(input_path)
    if not isinstance(payload, list):
        raise ValueError("Evaluation set JSON must be a list.")

    filled_case_id = 0
    filled_bucket = 0
    synced_tags = 0

    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            continue

        generated_case_id = f"{args.case_prefix}_{index + args.start_index:03d}"
        case_id = entry.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            entry["case_id"] = generated_case_id
            filled_case_id += 1

        bucket = entry.get("distribution_bucket")
        if not isinstance(bucket, str) or not bucket.strip():
            entry["distribution_bucket"] = _infer_bucket(entry)
            filled_bucket += 1

        has_query = entry.get("query") not in (None, "")
        tags = _normalized_tag_list(entry)
        next_tags = _sync_query_tags(tags, has_query=has_query)
        if next_tags != tags:
            entry["slice_tags"] = next_tags
            synced_tags += 1

    print(f"Entries: {len(payload)}")
    print(f"Filled case_id: {filled_case_id}")
    print(f"Filled distribution_bucket: {filled_bucket}")
    print(f"Synchronized query/no_query tags: {synced_tags}")

    if args.dry_run:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Wrote normalized evaluation set to {output_path}")


if __name__ == "__main__":
    main()
