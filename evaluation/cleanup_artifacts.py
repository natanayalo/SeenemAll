"""Move generated evaluation artifacts out of evaluation/ root."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


GENERATED_PATTERNS = (
    "baseline_*.md",
    "*_results.csv",
    "*_results_*.csv",
    "*_summary.json",
    "*_summary_*.json",
    "*_per_query_diff.json",
    "gate_report.json",
    "tmp_*.csv",
    "tmp_*.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean evaluation generated artifacts.")
    parser.add_argument(
        "--evaluation-dir",
        default="evaluation",
        help="Evaluation directory root.",
    )
    parser.add_argument(
        "--target-subdir",
        default="artifacts/inbox",
        help="Destination subdirectory under evaluation-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without moving files.",
    )
    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)
    target_dir = evaluation_dir / args.target_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    seen: set[Path] = set()
    for pattern in GENERATED_PATTERNS:
        for path in sorted(evaluation_dir.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            if not path.is_file():
                continue
            # Keep source files in place.
            if path.name in {
                "evaluation_set.json",
                "evaluate.py",
                "capture_baseline.py",
                "check_release_gate.py",
                "validate_evaluation_set.py",
                "find_golden_set.py",
            }:
                continue
            destination = target_dir / path.name
            if args.dry_run:
                print(f"[DRY-RUN] move {path} -> {destination}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(destination))
                print(f"Moved {path} -> {destination}")
            moved += 1

    print(f"Processed artifacts: {moved}")


if __name__ == "__main__":
    main()
