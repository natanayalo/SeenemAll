"""Enforce statement coverage for every changed API, ETL and evaluation module."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


def changed_modules(root: Path, base: str = "HEAD") -> list[str]:
    paths = set()
    for args in (
        ["diff", "--name-only", "-z", base],
        ["ls-files", "--others", "--exclude-standard", "-z"],
    ):
        result = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        )
        paths.update(result.stdout.split("\0"))
    return sorted(
        path
        for path in paths
        if path.endswith(".py")
        and path.split("/")[0] in {"api", "etl", "evaluation"}
        and (root / path).is_file()
    )


def check_coverage(
    report: dict[str, Any], paths: list[str], root: Path, minimum: float = 85.0
) -> list[str]:
    files = {
        str(Path(name.replace("\\", "/")).resolve()): data
        for name, data in report["files"].items()
    }
    failures = []
    for name in paths:
        data = files.get(str((root / name).resolve()))
        if data is None:
            failures.append(f"{name}: absent from coverage report")
            continue
        summary = data["summary"]
        covered, statements = summary["covered_lines"], summary["num_statements"]
        percent = 100.0 * covered / statements if statements else 100.0
        print(f"{name}: {covered}/{statements} statements ({percent:.8f}%)")
        # Compare the counts, never the rounded display percentage.
        if covered * 100 < minimum * statements:
            failures.append(f"{name}: {percent:.8f}% < {minimum}%")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path(".cache/coverage.json"))
    parser.add_argument("--base", default="HEAD")
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        root = Path.cwd()
        failures = check_coverage(report, changed_modules(root, args.base), root)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Coverage gate could not run: {exc}")
        return 1
    for failure in failures:
        print(f"FAIL: {failure}")
    return int(bool(failures))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
