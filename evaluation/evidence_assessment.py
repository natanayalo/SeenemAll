"""Real, uncached comparison of sufficiency decisions across evidence versions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluation.evidence import build_item_evidence, load_catalog_metadata
from evaluation.judge.ollama import discover_ollama_judges
from evaluation.models import JudgeInput, TypedId

CASES_PATH = Path("evaluation/fixtures/sufficiency_cases_v1.json")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def assess(output: Path, repeats: int = 2) -> dict[str, Any]:
    """Keep expected decisions outside model inputs; this never writes approval."""
    if not 1 <= repeats <= 3:
        raise ValueError("Assessment repeats must be between one and three")
    if output.exists():
        raise ValueError("Assessment reports are immutable; choose a new path")
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    jobs: list[Any] = []
    judges = {}
    for version in ("v2.2", "v2.3"):
        catalog = load_catalog_metadata(evidence_version=version)
        judge = discover_ollama_judges(None if version == "v2.2" else version)[
            "bespoke-nimble-9b"
        ]
        if not judge.is_available():
            raise RuntimeError("The pinned Nimble artifact is unavailable")
        judges[version] = judge.qualification_fingerprint()
        for case in cases:
            tid = TypedId.parse(case["typed_id"])
            inp = JudgeInput(case["query"], build_item_evidence(tid, catalog[str(tid)]))
            jobs.extend(
                (version, case, judge, inp, repeat) for repeat in range(repeats)
            )
    input_manifest = [
        {
            "version": version,
            "case_id": case["case_id"],
            "query": inp.query,
            "evidence_text": inp.evidence.to_evidence_text(),
            "input_sha256": inp.input_hash(),
            "repeat": repeat,
        }
        for version, case, _, inp, repeat in jobs
    ]
    # Freeze and report the inputs before inference; no production cache is read.
    manifest_path = output.with_suffix(".inputs.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as fp:
        json.dump(input_manifest, fp, indent=2, ensure_ascii=False)
        fp.write("\n")

    def run(job):
        version, case, judge, inp, repeat = job
        result = judge.judge_pair(inp)
        expected = case["expected_sufficient"][version]
        correct = (
            result.execution_status == "success"
            and result.evidence_sufficiency == expected
        )
        if correct and expected:
            correct = (
                case.get("minimum_grade", 0)
                <= result.grade
                <= case.get("maximum_grade", 3)
            )
        print(
            f"{version} {case['case_id']} repeat {repeat + 1}: sufficient={result.evidence_sufficiency}, correct={correct}",
            flush=True,
        )
        return {
            "version": version,
            "case_id": case["case_id"],
            "repeat": repeat,
            "expected_sufficient": expected,
            "correct": correct,
            "result": result.to_dict(),
        }

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, jobs))
    summaries = {}
    for version in judges:
        rows = [row for row in results if row["version"] == version]
        summaries[version] = {
            "requests": len(rows),
            "correct": sum(row["correct"] for row in rows),
            "false_abstentions": sum(
                row["expected_sufficient"] and not row["result"]["evidence_sufficiency"]
                for row in rows
            ),
            "false_sufficiency": sum(
                not row["expected_sufficient"] and row["result"]["evidence_sufficiency"]
                for row in rows
            ),
            "execution_failures": sum(
                row["result"]["execution_status"] != "success" for row in rows
            ),
        }
    report = {
        "scope": "diagnostic, not qualification or baseline promotion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repeats": repeats,
        "case_count": len(cases),
        "native_requests": len(results),
        "production_cache_used": False,
        "case_manifest_sha256": fingerprint(cases),
        "input_manifest_sha256": fingerprint(input_manifest),
        "judge_fingerprints": judges,
        "summary": summaries,
        "results": results,
    }
    with output.open("x", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)
        fp.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    result = assess(args.output, args.repeats)
    print(json.dumps(result["summary"], indent=2))
    return int(
        any(
            summary["execution_failures"] or summary["correct"] != summary["requests"]
            for summary in result["summary"].values()
        )
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
