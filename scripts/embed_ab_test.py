"""Utilities for offline A/B testing of embedding strategies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sqlalchemy.orm import Session

from api.db.models import ItemEmbedding
from api.db.session import get_sessionmaker
from scripts.evaluate_models import load_test_pairs


@dataclass
class PairComparison:
    """Stores per-pair comparison metrics for two embedding versions."""

    item_a: int
    item_b: int
    expected: float
    score_a: float
    score_b: float
    abs_error_a: float
    abs_error_b: float


@dataclass
class ABReport:
    """Aggregate comparison report for two embedding versions."""

    version_a: str
    version_b: str
    mae_a: float
    mae_b: float
    win_rate_b: float
    tie_rate: float
    pair_results: List[PairComparison]

    def to_dict(self) -> Dict[str, object]:
        """Convert report to plain dict for serialization or logging."""
        data = asdict(self)
        data["pair_results"] = [asdict(pr) for pr in self.pair_results]
        return data


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Safe cosine similarity that tolerates zero vectors."""
    vec_a = np.asarray(a, dtype=np.float32)
    vec_b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def load_embeddings(
    session: Session, version: str, item_ids: Iterable[int]
) -> Dict[int, Sequence[float]]:
    """Fetch embeddings for the given version and item ids."""
    ids = sorted(set(item_ids))
    if not ids:
        return {}

    rows = session.query(ItemEmbedding).all()
    embeddings: Dict[int, Sequence[float]] = {}
    for row in rows:
        if row.version == version and row.item_id in ids:
            embeddings[row.item_id] = row.vector

    missing = set(ids) - set(embeddings)
    if missing:
        raise ValueError(
            f"Missing embeddings for version '{version}': {sorted(missing)}"
        )
    return embeddings


def compare_versions(
    embeddings_a: Dict[int, Sequence[float]],
    embeddings_b: Dict[int, Sequence[float]],
    pairs: Sequence[Tuple[int, int, float]],
    version_a: str,
    version_b: str,
) -> ABReport:
    """Compute comparison metrics for two embedding versions."""
    if not pairs:
        raise ValueError("At least one test pair is required for A/B comparison")

    pair_results: List[PairComparison] = []
    total_err_a = 0.0
    total_err_b = 0.0
    wins_b = 0
    ties = 0

    for item_a, item_b, expected in pairs:
        if item_a not in embeddings_a or item_b not in embeddings_a:
            raise KeyError(f"Item {item_a} or {item_b} missing in embeddings for {version_a}")
        if item_a not in embeddings_b or item_b not in embeddings_b:
            raise KeyError(f"Item {item_a} or {item_b} missing in embeddings for {version_b}")

        score_a = _cosine(embeddings_a[item_a], embeddings_a[item_b])
        score_b = _cosine(embeddings_b[item_a], embeddings_b[item_b])

        err_a = abs(score_a - expected)
        err_b = abs(score_b - expected)

        total_err_a += err_a
        total_err_b += err_b

        if err_b < err_a:
            wins_b += 1
        elif err_b == err_a:
            ties += 1

        pair_results.append(
            PairComparison(
                item_a=item_a,
                item_b=item_b,
                expected=expected,
                score_a=score_a,
                score_b=score_b,
                abs_error_a=err_a,
                abs_error_b=err_b,
            )
        )

    n_pairs = len(pairs)
    mae_a = total_err_a / n_pairs
    mae_b = total_err_b / n_pairs
    win_rate_b = wins_b / n_pairs
    tie_rate = ties / n_pairs

    return ABReport(
        version_a=version_a,
        version_b=version_b,
        mae_a=mae_a,
        mae_b=mae_b,
        win_rate_b=win_rate_b,
        tie_rate=tie_rate,
        pair_results=pair_results,
    )


def run_ab_test(
    session: Session, version_a: str, version_b: str
) -> ABReport:
    """Load embeddings and test pairs, then run the comparison."""
    pairs = load_test_pairs()
    item_ids = {item for pair in pairs for item in pair[:2]}

    embeddings_a = load_embeddings(session, version_a, item_ids)
    embeddings_b = load_embeddings(session, version_b, item_ids)

    return compare_versions(embeddings_a, embeddings_b, pairs, version_a, version_b)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B test embedding versions.")
    parser.add_argument(
        "--version-a",
        default="v1",
        help="Baseline embedding version to compare.",
    )
    parser.add_argument(
        "--version-b",
        required=True,
        help="Candidate embedding version to compare.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI wrapper
    args = _parse_args(argv)
    SessionLocal = get_sessionmaker()
    with SessionLocal() as session:
        report = run_ab_test(session, args.version_a, args.version_b)
        print(f"Baseline ({args.version_a}) MAE: {report.mae_a:.4f}")
        print(f"Candidate ({args.version_b}) MAE: {report.mae_b:.4f}")
        print(f"Win rate (candidate better): {report.win_rate_b * 100:.1f}%")
        print(f"Tie rate: {report.tie_rate * 100:.1f}%")


if __name__ == "__main__":  # pragma: no cover
    main()
