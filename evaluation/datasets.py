"""Public and local dataset loaders for Seen'emAll Evaluation Suite v2."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict

import numpy as np

from evaluation.evidence import index_catalog_metadata

from evaluation.models import (
    DeterministicConstraint,
    TestCase,
    TypedId,
)

DATASETS_DIR = Path("evaluation/datasets")
FIXTURES_DIR = Path("evaluation/fixtures")


class PublicDatasetEntry(TypedDict, total=False):
    query: str
    golden_ids: List[int]
    user_id: Optional[str]
    train_items: Optional[List[Dict[str, Any]]]
    seed_history: Optional[List[Dict[str, Any]]]
    persona_context: Optional[Dict[str, Any]]
    golden_scores: Optional[Dict[int, float]]
    golden_set: Optional[List[Dict[str, Any]]]
    raw: Optional[Dict[str, Any]]


def load_evaluation_cases(
    track: str = "product",
    split: str = "dev",
    base_dir: Optional[Path] = None,
    path: Optional[Path] = None,
) -> List[TestCase]:
    """Load test cases from unified dataset files or explicit path."""
    if path is None and (track.lower(), split.lower()) == ("product", "full"):
        d_dir = base_dir or DATASETS_DIR
        if not all(
            (d_dir / name).exists() for name in ("product_dev.json", "product_reg.json")
        ):
            return []
        return load_evaluation_cases(track, "dev", d_dir) + load_evaluation_cases(
            track, "regression", d_dir
        )
    target_file: Optional[Path] = None
    if path is not None:
        target_file = path
    else:
        d_dir = base_dir or DATASETS_DIR
        filename_map = {
            ("product", "dev"): d_dir / "product_dev.json",
            ("product", "regression"): d_dir / "product_reg.json",
            ("product", "reg"): d_dir / "product_reg.json",
            ("cold_start", "dev"): d_dir / "cold_start.json",
            ("cold_start", "regression"): d_dir / "cold_start.json",
            ("cold_start", "full"): d_dir / "cold_start.json",
        }
        target_file = filename_map.get((track.lower(), split.lower()))

    if target_file is None or not target_file.exists():
        # Fallback to legacy evaluation_set.json if split file does not exist
        legacy_file = Path("evaluation/evaluation_set.json")
        if legacy_file.exists():
            return _load_legacy_cases(legacy_file, track=track)
        return []

    with target_file.open("r", encoding="utf-8") as fp:
        raw_items = json.load(fp)

    cases: List[TestCase] = []
    for item in raw_items:
        constraints_data = item.get("constraints")
        constraints = None
        if isinstance(constraints_data, dict):
            constraints = DeterministicConstraint(
                media_type=constraints_data.get("media_type"),
                min_year=constraints_data.get("min_year")
                or constraints_data.get("year_gte"),
                max_year=constraints_data.get("max_year")
                or constraints_data.get("year_lte"),
                min_runtime=constraints_data.get("min_runtime"),
                max_runtime=constraints_data.get("max_runtime"),
                language=constraints_data.get("language"),
                providers=(
                    constraints_data.get("providers")
                    if isinstance(constraints_data.get("providers"), list)
                    else (
                        [constraints_data["providers"]]
                        if constraints_data.get("providers")
                        else None
                    )
                ),
                genres=(
                    constraints_data.get("genres")
                    if isinstance(constraints_data.get("genres"), list)
                    else (
                        [constraints_data["genres"]]
                        if constraints_data.get("genres")
                        else None
                    )
                ),
                require_all_genres=bool(constraints_data.get("require_all_genres")),
                seen_ids=constraints_data.get("seen_ids"),
                disliked_ids=constraints_data.get("disliked_ids"),
                canonical_sequence_id=constraints_data.get("canonical_sequence_id"),
            )

        default_media = (
            item.get("media_type")
            or (constraints.media_type if constraints else None)
            or "movie"
        )
        raw_goldens = item.get("golden_set", item.get("golden_ids", []))
        typed_goldens = [TypedId.parse(g, default_media) for g in raw_goldens]
        golden_ids = [tid.id for tid in typed_goldens]
        golden_set = [
            {**g, "media_type": tid.media_type} if isinstance(g, dict) else str(tid)
            for g, tid in zip(raw_goldens, typed_goldens)
        ]

        eligible_count = item.get("eligible_catalog_count")
        if eligible_count is None:
            if item.get("canonical_sequence"):
                eligible_count = len(item["canonical_sequence"])
            elif bool(item.get("expected_empty", False)):
                eligible_count = 0
            elif item.get("task") == "franchise" and golden_ids:
                eligible_count = len(golden_ids)

        cases.append(
            TestCase(
                case_id=item.get("case_id", f"case_{len(cases)+1}"),
                family_id=item.get("family_id", f"fam_{len(cases)+1}"),
                track=item.get("track", track),
                split=item.get("split", split),
                task=item.get("task", "search"),
                slice_tags=item.get("slice_tags", [item.get("category", "general")]),
                query=item["query"],
                user_id=item.get("user_id", "u1"),
                persona_context=item.get("persona_context"),
                constraints=constraints,
                golden_ids=golden_ids,
                golden_set=golden_set,
                canonical_sequence=item.get("canonical_sequence"),
                expected_empty=bool(item.get("expected_empty", False)),
                eligible_catalog_count=eligible_count,
            )
        )
    return cases


def _load_legacy_cases(legacy_path: Path, track: str = "product") -> List[TestCase]:
    with legacy_path.open("r", encoding="utf-8") as fp:
        raw_items = json.load(fp)
    cases: List[TestCase] = []
    for idx, item in enumerate(raw_items):
        cat = item.get("category", "general")
        if track == "cold_start" and cat != "cold_start":
            continue
        if track == "product" and cat == "cold_start":
            continue

        raw_goldens = item.get("golden_set", [])
        typed_goldens = [
            TypedId.parse(g, item.get("media_type") or "movie") for g in raw_goldens
        ]
        golden = [tid.id for tid in typed_goldens]
        golden_set = [
            {**g, "media_type": tid.media_type} if isinstance(g, dict) else str(tid)
            for g, tid in zip(raw_goldens, typed_goldens)
        ]
        eligible_count = None
        if item.get("canonical_sequence"):
            eligible_count = len(item["canonical_sequence"])
        elif item.get("expected_empty", False):
            eligible_count = 0
        elif cat == "franchise" and golden:
            eligible_count = len(golden)

        cases.append(
            TestCase(
                case_id=f"legacy_{idx+1}",
                family_id=f"legacy_fam_{idx+1}",
                track="cold_start" if cat == "cold_start" else "product",
                split="dev",
                task="search",
                slice_tags=[cat],
                query=item["query"],
                user_id=item.get("user_id", "u1"),
                golden_ids=golden,
                golden_set=golden_set,
                eligible_catalog_count=eligible_count,
            )
        )
    return cases


def load_public_dataset(name: str) -> List[PublicDatasetEntry]:
    """Return evaluation entries sourced from a public external dataset."""
    normalized = name.lower()
    if normalized == "movielens20m":
        return MovieLens20MLoader().load_as_public_entries()
    if normalized == "tag_genome":
        return TagGenomeLoader().load_as_public_entries()
    raise NotImplementedError(
        f"Unknown public dataset '{name}'. Available: movielens20m, tag_genome"
    )


class MovieLens20MLoader:
    """MovieLens 20M temporal benchmark loader.

    Enforces temporal boundaries:
      - Training before April 1, 2014 (< 1396310400 UTC).
      - Validation from April 1 through September 30, 2014.
      - Test from October 1, 2014 through March 31, 2015 (all UTC).
      - Mapped catalog release year <= 2013.
      - Ratings mapping:
          4.5-5.0 -> Grade 3
          4.0     -> Grade 2
          3.0-3.5 -> Grade 1
          < 3.0   -> Grade 0
      - Eligibility: >= 20 training interactions, >= 5 test positives per user.
      - Sample 500 eligible users across training-history quartiles with seed 42.
    """

    TRAIN_CUTOFF = datetime(2014, 4, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    VAL_CUTOFF = datetime(2014, 10, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    TEST_CUTOFF = datetime(2015, 4, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()

    @staticmethod
    def map_rating_to_grade(rating: float) -> int:
        if rating >= 4.5:
            return 3
        if rating >= 4.0:
            return 2
        if rating >= 3.0:
            return 1
        return 0

    def load_as_public_entries(
        self,
        interactions_path: Optional[Path] = None,
        movie_catalog_path: Optional[Path] = None,
    ) -> List[PublicDatasetEntry]:
        """Load sampled eligible MovieLens 20M users as PublicDatasetEntry query-gold pairs."""
        path = interactions_path or (FIXTURES_DIR / "movielens_sample.json")
        if not path.exists():
            raise FileNotFoundError(
                f"MovieLens 20M dataset file not found at '{path}'. "
                "Missing external benchmark data produces an explicit unavailable result; "
                "fabricated samples must never satisfy benchmark gates."
            )

        corrected_catalog = FIXTURES_DIR / "catalog_evidence_v2.2.json"
        cat_path = movie_catalog_path or (
            corrected_catalog
            if corrected_catalog.exists()
            else FIXTURES_DIR / "catalog_metadata.json"
        )
        if not cat_path.exists():
            raise FileNotFoundError(
                f"Candidate catalog universe file not found at '{cat_path}'. "
                "External benchmarks must be strictly grounded against the candidate catalog."
            )

        try:
            with cat_path.open("r", encoding="utf-8") as fp:
                cat_data = json.load(fp)
                if isinstance(cat_data, dict):
                    # MovieLens declares movie-only IDs for legacy bare catalogs;
                    # explicit TV rows and typed keys must retain their namespace.
                    cat_data = {
                        key: {
                            **row,
                            "media_type": row.get("media_type")
                            or TypedId.parse(key).media_type,
                        }
                        for key, row in cat_data.items()
                    }
                elif isinstance(cat_data, list):
                    cat_data = [
                        {**row, "media_type": row.get("media_type") or "movie"}
                        for row in cat_data
                    ]
                else:
                    cat_data = []
                catalog_map = {
                    TypedId.parse(key).id: row
                    for key, row in index_catalog_metadata(cat_data).items()
                    if row["media_type"] == "movie"
                }
        except Exception as exc:
            raise ValueError(
                f"Failed to parse candidate catalog universe at '{cat_path}': {exc}"
            ) from exc

        if not catalog_map:
            raise ValueError(
                f"Candidate catalog universe at '{cat_path}' is empty or invalid. "
                "Grounding requires a non-empty candidate catalog."
            )

        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)

        raw_users = data.get("users", [])
        eligible_users = []

        for u in raw_users:
            raw_train = u.get("train_items", []) or u.get("seed_history", [])
            for it in raw_train:
                if it.get("timestamp") is None:
                    raise ValueError(
                        f"Null or empty timestamp in user interactions: user {u.get('user_id')}"
                    )
            train_items = [
                it for it in raw_train if it["timestamp"] < self.TRAIN_CUTOFF
            ]
            train_seen_ids = {
                int(it["tmdb_id"]) for it in train_items if "tmdb_id" in it
            }

            raw_test = u.get("test_items", [])
            for it in raw_test:
                if it.get("timestamp") is None:
                    raise ValueError(
                        f"Null or empty timestamp in user test interactions: user {u.get('user_id')}"
                    )
            test_items = [
                it
                for it in raw_test
                if self.VAL_CUTOFF <= it["timestamp"] < self.TEST_CUTOFF
            ]

            # Enforce catalog release year <= 2013 and presence in frozen candidate catalog
            valid_test_items = []
            for it in test_items:
                tmdb_id = int(it.get("tmdb_id", 0))
                if tmdb_id in train_seen_ids:
                    # Exclude already-seen items from test positives!
                    continue

                if tmdb_id not in catalog_map:
                    # Strictly filter positives to enforce candidate universe membership
                    continue

                rel_year = catalog_map[tmdb_id].get(
                    "release_year", it.get("release_year")
                )
                if rel_year is not None and rel_year <= 2013:
                    valid_test_items.append(it)

            test_positives = [
                int(it["tmdb_id"])
                for it in valid_test_items
                if self.map_rating_to_grade(it.get("rating", 0.0)) >= 2
            ]

            # Eligibility: >= 20 training interactions, >= 5 test positives per user
            train_count = len(train_items)
            if train_count >= 20 and len(test_positives) >= 5:
                eligible_users.append((u, test_positives, train_items, train_count))

        # Sample up to 500 eligible users across training-history quartiles with seed 42
        if len(eligible_users) > 500:
            eligible_users.sort(key=lambda x: x[3])
            indices = np.linspace(0, len(eligible_users) - 1, 500, dtype=int)
            eligible_users = [eligible_users[i] for i in indices]

        entries: List[PublicDatasetEntry] = []
        for u, test_positives, train_items, _ in eligible_users:
            u_id = str(u.get("user_id", "u1"))
            entries.append(
                {
                    "query": f"MovieLens user {u_id} history profile",
                    "user_id": u_id,
                    "train_items": train_items,
                    "seed_history": train_items,
                    "persona_context": {
                        "user_id": u_id,
                        "train_items": train_items,
                        "seed_history": train_items,
                    },
                    "golden_ids": test_positives,
                }
            )
        return entries


class TagGenomeLoader:
    """Tag Genome loader with single-tag queries and continuous identity gain."""

    DEFAULT_TAGS = [
        "atmospheric",
        "cyberpunk",
        "dystopian",
        "dark comedy",
        "time travel",
        "artificial intelligence",
        "post-apocalyptic",
        "heist",
        "superhero",
        "martial arts",
    ]

    def __init__(self, genome_path: Optional[Path] = None) -> None:
        self.genome_path = genome_path or (FIXTURES_DIR / "tag_genome.json")

    def load_as_public_entries(
        self,
        tags: Optional[Sequence[str]] = None,
    ) -> List[PublicDatasetEntry]:
        """Load single-tag queries over genome-covered catalog with continuous scores."""
        if not self.genome_path.exists():
            raise FileNotFoundError(
                f"Tag Genome dataset file not found at '{self.genome_path}'. "
                "Missing external benchmark data produces an explicit unavailable result; "
                "fixed placeholder title lists must never satisfy benchmark gates."
            )

        with self.genome_path.open("r", encoding="utf-8") as fp:
            genome_data: Dict[str, Dict[str, float]] = json.load(fp)

        selected_tags = tags or self.DEFAULT_TAGS
        entries: List[PublicDatasetEntry] = []

        for tag in selected_tags:
            tag_scores = genome_data.get(tag, {})
            # Continuous relevance scores: score in [0.0, 1.0]
            sorted_items = sorted(
                tag_scores.items(), key=lambda kv: float(kv[1]), reverse=True
            )
            golden_scores = {
                int(tmdb_id): float(score) for tmdb_id, score in sorted_items
            }
            golden_set = [
                {
                    "id": int(tmdb_id),
                    "score": float(score),
                    "relevance": float(score),
                }
                for tmdb_id, score in sorted_items
            ]
            golden_ids = [
                int(tmdb_id) for tmdb_id, score in sorted_items if float(score) >= 0.5
            ]
            entries.append(
                {
                    "query": tag,
                    "golden_ids": golden_ids,
                    "golden_scores": golden_scores,
                    "golden_set": golden_set,
                }
            )
        return entries
