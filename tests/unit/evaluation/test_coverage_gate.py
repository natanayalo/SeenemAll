import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation import coverage_gate


def test_unrounded_per_module_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = {
        "files": {
            "api\\a.py": {"summary": {"covered_lines": 844, "num_statements": 994}},
            "etl/empty.py": {"summary": {"covered_lines": 0, "num_statements": 0}},
            "evaluation/pass.py": {
                "summary": {"covered_lines": 85, "num_statements": 100}
            },
        }
    }
    failures = coverage_gate.check_coverage(
        report,
        ["api/a.py", "etl/empty.py", "evaluation/pass.py", "api/missing.py"],
        tmp_path,
    )
    assert len(failures) == 2
    assert "84.90945674" in failures[0]  # A displayed 85% is below the threshold.
    assert "absent" in failures[1]


def test_git_change_discovery(tmp_path, monkeypatch):
    for name in ["api/a.py", "evaluation/new.py", "tests/test_a.py"]:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    outputs = iter(
        ["api/a.py\0api/deleted.py\0tests/test_a.py\0", "evaluation/new.py\0"]
    )
    monkeypatch.setattr(
        coverage_gate.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=next(outputs)),
    )
    assert coverage_gate.changed_modules(tmp_path, "base") == [
        "api/a.py",
        "evaluation/new.py",
    ]


@pytest.mark.parametrize("mode", ["success", "below", "invalid", "missing"])
def test_coverage_cli(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    report = Path("coverage.json")
    monkeypatch.setattr(coverage_gate, "changed_modules", lambda *a: ["api/a.py"])
    if mode == "invalid":
        report.write_text("broken")
    elif mode != "missing":
        report.write_text(
            json.dumps(
                {
                    "files": {
                        "api/a.py": {
                            "summary": {
                                "covered_lines": 85 if mode == "success" else 84,
                                "num_statements": 100,
                            }
                        }
                    }
                }
            )
        )
    assert coverage_gate.main(["--report", str(report)]) == (
        0 if mode == "success" else 1
    )
