"""Validate offline evaluation set labels against local catalog data."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

VALID_MEDIA_TYPES = {"movie", "tv"}
RUNTIME_CAP_RE = re.compile(r"under\s+(\d+)\s+minutes", re.IGNORECASE)


def _load_evaluation_set(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Evaluation set must be a JSON list.")
    return payload


def _runtime_cap(query: str | None) -> int | None:
    if not query:
        return None
    match = RUNTIME_CAP_RE.search(query)
    if not match:
        return None
    return int(match.group(1))


def _validate_structure(
    evaluation_set: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], Dict[int, set[str]]]:
    errors: List[str] = []
    warnings: List[str] = []
    id_to_media: Dict[int, set[str]] = defaultdict(set)
    seen_case_ids: set[str] = set()

    for index, entry in enumerate(evaluation_set):
        location = f"entry[{index}]"
        case_id = entry.get("case_id")
        if isinstance(case_id, str) and case_id.strip():
            normalized = case_id.strip()
            if normalized in seen_case_ids:
                errors.append(f"{location}: duplicate case_id '{normalized}'.")
            seen_case_ids.add(normalized)

        golden_set = entry.get("golden_set")
        if not isinstance(golden_set, list) or not golden_set:
            errors.append(f"{location}: missing or empty golden_set list.")
            continue

        seen_in_case: set[Tuple[int, str]] = set()
        positive_keys: set[Tuple[int, str]] = set()
        for pos, item in enumerate(golden_set):
            item_loc = f"{location}.golden_set[{pos}]"
            if not isinstance(item, dict):
                errors.append(f"{item_loc}: item must be an object.")
                continue

            try:
                tmdb_id = int(item.get("id"))
            except (TypeError, ValueError):
                errors.append(f"{item_loc}: invalid id '{item.get('id')}'.")
                continue

            title = item.get("title")
            if not isinstance(title, str) or not title.strip():
                errors.append(f"{item_loc}: missing/empty title.")

            media_type_raw = item.get("media_type")
            media_type = (
                str(media_type_raw).strip().lower()
                if isinstance(media_type_raw, str)
                else ""
            )
            if media_type not in VALID_MEDIA_TYPES:
                errors.append(
                    f"{item_loc}: media_type must be one of {sorted(VALID_MEDIA_TYPES)}, "
                    f"got '{media_type_raw}'."
                )
                continue

            key = (tmdb_id, media_type)
            if key in seen_in_case:
                errors.append(f"{item_loc}: duplicate golden tuple {key} in case.")
            seen_in_case.add(key)
            positive_keys.add(key)
            id_to_media[tmdb_id].add(media_type)

            if "relevance" in item:
                try:
                    relevance = float(item["relevance"])
                except (TypeError, ValueError):
                    errors.append(f"{item_loc}: invalid relevance '{item['relevance']}'.")
                else:
                    if relevance < 0.0 or relevance > 3.0:
                        errors.append(
                            f"{item_loc}: relevance out of range [0,3], got {relevance}."
                        )

        negative_set = entry.get("negative_set")
        if negative_set is not None:
            if not isinstance(negative_set, list):
                errors.append(f"{location}: negative_set must be a list when provided.")
            else:
                seen_negative: set[Tuple[int, str]] = set()
                for neg_pos, item in enumerate(negative_set):
                    neg_loc = f"{location}.negative_set[{neg_pos}]"
                    if not isinstance(item, dict):
                        errors.append(f"{neg_loc}: item must be an object.")
                        continue
                    try:
                        neg_id = int(item.get("id"))
                    except (TypeError, ValueError):
                        errors.append(f"{neg_loc}: invalid id '{item.get('id')}'.")
                        continue
                    media_raw = item.get("media_type")
                    media_type = (
                        str(media_raw).strip().lower() if isinstance(media_raw, str) else ""
                    )
                    if media_type not in VALID_MEDIA_TYPES:
                        errors.append(
                            f"{neg_loc}: media_type must be one of {sorted(VALID_MEDIA_TYPES)}, "
                            f"got '{media_raw}'."
                        )
                        continue
                    key = (neg_id, media_type)
                    if key in seen_negative:
                        errors.append(f"{neg_loc}: duplicate negative tuple {key} in case.")
                    seen_negative.add(key)
                    id_to_media[neg_id].add(media_type)
                    if key in positive_keys:
                        errors.append(
                            f"{neg_loc}: tuple {key} appears in both golden_set and negative_set."
                        )

    for tmdb_id, media_types in sorted(id_to_media.items()):
        if len(media_types) > 1:
            warnings.append(
                "dataset-level tmdb_id collision across media types: "
                f"id={tmdb_id} media_types={sorted(media_types)}"
            )

    return errors, warnings, id_to_media


def _validate_catalog_alignment(
    evaluation_set: List[Dict[str, Any]],
    *,
    database_url: str,
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    ids = sorted(
        {
            int(item["id"])
            for entry in evaluation_set
            for key in ("golden_set", "negative_set")
            for item in entry.get(key, [])
            if isinstance(item, dict) and "id" in item
        }
    )
    if not ids:
        return errors, warnings

    lookup: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    try:
        engine = create_engine(database_url, future=True)
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    "select tmdb_id, media_type, title, runtime "
                    "from items "
                    "where tmdb_id = any(:ids)"
                ),
                {"ids": ids},
            ).mappings()
            for row in rows:
                key = (int(row["tmdb_id"]), str(row["media_type"]).strip().lower())
                lookup[key].append(
                    {
                        "title": str(row["title"]),
                        "runtime": row["runtime"],
                    }
                )
    except SQLAlchemyError as exc:
        errors.append(f"catalog check failed: could not query database ({exc}).")
        return errors, warnings

    for index, entry in enumerate(evaluation_set):
        query = entry.get("query")
        cap = _runtime_cap(query if isinstance(query, str) else None)
        for set_name in ("golden_set", "negative_set"):
            for pos, item in enumerate(entry.get(set_name, [])):
                if not isinstance(item, dict):
                    continue
                media_type = str(item.get("media_type", "")).strip().lower()
                if media_type not in VALID_MEDIA_TYPES:
                    continue
                try:
                    tmdb_id = int(item["id"])
                except (TypeError, ValueError):
                    continue
                title = str(item.get("title", ""))
                location = f"entry[{index}].{set_name}[{pos}]"
                key = (tmdb_id, media_type)
                catalog_rows = lookup.get(key, [])
                if not catalog_rows:
                    errors.append(
                        f"{location}: missing catalog tuple for id={tmdb_id}, media_type={media_type}."
                    )
                    continue

                if title:
                    catalog_titles = {row["title"] for row in catalog_rows}
                    if title not in catalog_titles:
                        errors.append(
                            f"{location}: title mismatch for id={tmdb_id}, media_type={media_type}. "
                            f"golden='{title}' catalog={sorted(catalog_titles)}"
                        )

                # Runtime caps only apply to positive golden labels.
                if set_name != "golden_set" or cap is None:
                    continue
                runtimes = [
                    int(row["runtime"])
                    for row in catalog_rows
                    if row.get("runtime") is not None
                ]
                if runtimes and min(runtimes) > cap:
                    errors.append(
                        f"{location}: runtime cap violation for query '{query}'. "
                        f"runtime(min)={min(runtimes)} > {cap}"
                    )
                elif not runtimes:
                    warnings.append(
                        f"{location}: runtime unknown in catalog; cannot verify cap {cap}."
                    )

    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate evaluation_set.json.")
    parser.add_argument(
        "--evaluation-set",
        default="evaluation/evaluation_set.json",
        help="Path to evaluation set JSON.",
    )
    parser.add_argument(
        "--database-url",
        default="",
        help="Optional SQLAlchemy database URL for catalog checks.",
    )
    args = parser.parse_args()

    load_dotenv()

    evaluation_path = Path(args.evaluation_set)
    evaluation_set = _load_evaluation_set(evaluation_path)

    errors, warnings, id_to_media = _validate_structure(evaluation_set)

    database_url = args.database_url or os.getenv("DATABASE_URL", "")
    if database_url:
        # If running from host with docker-compose defaults, use localhost for DB.
        if "@db:" in database_url:
            database_url = database_url.replace("@db:", "@localhost:")
        db_errors, db_warnings = _validate_catalog_alignment(
            evaluation_set,
            database_url=database_url,
        )
        errors.extend(db_errors)
        warnings.extend(db_warnings)
    else:
        warnings.append("No DATABASE_URL provided; skipped catalog alignment checks.")

    print(f"Checked entries: {len(evaluation_set)}")
    print(f"Unique golden tmdb_ids: {len(id_to_media)}")
    print(f"Warnings: {len(warnings)}")
    print(f"Errors: {len(errors)}")

    for warning in warnings:
        print(f"[WARN] {warning}")
    for err in errors:
        print(f"[ERROR] {err}")

    if errors:
        sys.exit(1)
    print("Validation passed.")


if __name__ == "__main__":
    main()
