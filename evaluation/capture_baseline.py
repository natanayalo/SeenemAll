"""Capture a dated recommendation-quality baseline snapshot."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    httpx = None

_MAX_COMMAND_OUTPUT_CHARS = 12000


def _load_evaluation_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _truncate_output(text: str, max_chars: int = _MAX_COMMAND_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    hidden = len(text) - max_chars
    return f"{text[:max_chars]}\n...[truncated {hidden} chars]"


def _run_command(command: str, timeout_s: float) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "exit_code": int(completed.returncode),
            "stdout": _truncate_output(completed.stdout.strip()),
            "stderr": _truncate_output(completed.stderr.strip()),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "exit_code": -1,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": f"Timeout after {timeout_s}s",
        }


def _load_debug_samples(path: Path, limit: int) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        evaluation_set = json.load(handle)

    samples: List[Dict[str, Any]] = []
    for entry in evaluation_set:
        if len(samples) >= limit:
            break
        request_overrides = dict(entry.get("request_overrides") or {})
        sample: Dict[str, Any] = {
            "user_id": request_overrides.get("user_id", "u1"),
            "limit": 5,
        }
        query = entry.get("query")
        if query not in (None, ""):
            sample["query"] = str(query)
        profile = request_overrides.get("profile")
        if profile:
            sample["profile"] = str(profile)
        samples.append(sample)

    return samples


def _resolve_default_commands(
    snapshot_date: str, config_path: Path
) -> tuple[str, str, str]:
    if shutil.which("make"):
        return f'make eval EVAL_CONFIG="{config_path}"', "make metrics", "make"

    venv_python = Path(".venv") / "Scripts" / "python.exe"
    if venv_python.exists():
        python_cmd = f'"{venv_python}"'
    else:
        python_cmd = "python"

    eval_command = (
        f"{python_cmd} evaluation/evaluate.py "
        f'--config "{config_path}" '
        f"--output-csv evaluation/artifacts/snapshots/evaluation_results_{snapshot_date}.csv "
        f"--summary-json evaluation/artifacts/snapshots/evaluation_summary_{snapshot_date}.json"
    )
    metrics_command = "curl -s http://localhost:8000/healthz/metrics"
    return eval_command, metrics_command, "fallback_no_make"


def _run_debug_request(
    base_url: str, params: Dict[str, Any], timeout_s: float
) -> Dict[str, Any]:
    url = f"{base_url}?{urlencode(params)}"

    def _summarize(payload: Dict[str, Any]) -> Dict[str, Any]:
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            items = []
        return {
            "items_count": len(items),
            "top_ids": [item.get("id") for item in items[:5] if isinstance(item, dict)],
            "top_tmdb_ids": [
                item.get("tmdb_id") for item in items[:5] if isinstance(item, dict)
            ],
            "debug_metrics": (
                (payload.get("debug") or {}).get("metrics")
                if isinstance(payload, dict)
                else None
            ),
            "raw_error": payload.get("error") if isinstance(payload, dict) else None,
        }

    if httpx is None:
        try:
            with urlopen(url, timeout=timeout_s) as response:
                status = int(getattr(response, "status", 200))
                raw = response.read().decode("utf-8")
            payload = json.loads(raw) if raw else {}
            return {
                "url": url,
                "status_code": status,
                "ok": 200 <= status < 300,
                "payload": _summarize(payload),
            }
        except HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode("utf-8"))
            except Exception:
                payload = {"error": str(exc)}
            return {
                "url": url,
                "status_code": int(exc.code),
                "ok": False,
                "payload": _summarize(payload),
            }
        except URLError as exc:
            return {
                "url": url,
                "status_code": None,
                "ok": False,
                "payload": {"error": str(exc)},
            }
        except Exception as exc:
            return {
                "url": url,
                "status_code": None,
                "ok": False,
                "payload": {"error": str(exc)},
            }

    try:
        response = httpx.get(base_url, params=params, timeout=timeout_s)
        payload = response.json() if response.text else {}
        return {
            "url": url,
            "status_code": int(response.status_code),
            "ok": bool(response.is_success),
            "payload": _summarize(payload),
        }
    except Exception as exc:
        return {
            "url": url,
            "status_code": None,
            "ok": False,
            "payload": {"error": str(exc)},
        }


def _render_command_section(result: Dict[str, Any]) -> str:
    lines = [
        f"### `{result['command']}`",
        f"- Exit code: `{result['exit_code']}`",
        "",
        "```text",
        result.get("stdout") or "<no stdout>",
        "```",
    ]
    stderr = result.get("stderr")
    if stderr:
        lines.extend(["", "```text", stderr, "```"])
    return "\n".join(lines)


def _render_debug_section(results: List[Dict[str, Any]]) -> str:
    lines = ["## 20 Sampled `/recommend/debug` Requests", ""]
    for idx, result in enumerate(results, start=1):
        lines.append(f"### Request {idx}")
        lines.append(f"- URL: `{result['url']}`")
        lines.append(f"- Status: `{result['status_code']}`")
        lines.append(f"- OK: `{result['ok']}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(result.get("payload", {}), indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture quality baseline snapshot.")
    parser.add_argument(
        "--config",
        default="evaluation/evaluation_config.json",
        help="Optional evaluator config JSON path.",
    )
    parser.add_argument(
        "--eval-set",
        default="",
        help=(
            "Path to evaluation set for debug query sampling. "
            "Defaults to config.evaluation_set or evaluation/evaluation_set_v2.json."
        ),
    )
    parser.add_argument(
        "--debug-url",
        default="http://localhost:8000/recommend/debug",
        help="Debug endpoint URL.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path. Defaults to evaluation/artifacts/baselines/baseline_<date>.md",
    )
    parser.add_argument(
        "--command-timeout-s",
        type=float,
        default=180.0,
        help="Timeout for each shell command.",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=8.0,
        help="Timeout for each debug request.",
    )
    parser.add_argument(
        "--eval-command",
        default="",
        help="Optional command override for the eval step.",
    )
    parser.add_argument(
        "--metrics-command",
        default="",
        help="Optional command override for the metrics step.",
    )
    args = parser.parse_args()

    snapshot_date = dt.date.today().isoformat()
    output_path = (
        Path(args.output)
        if args.output
        else Path("evaluation")
        / "artifacts"
        / "baselines"
        / f"baseline_{snapshot_date}.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    config = _load_evaluation_config(config_path)

    default_eval_command, default_metrics_command, command_mode = (
        _resolve_default_commands(snapshot_date, config_path)
    )
    eval_command = args.eval_command or default_eval_command
    metrics_command = args.metrics_command or default_metrics_command

    command_results = [
        _run_command(eval_command, args.command_timeout_s),
        _run_command(metrics_command, args.command_timeout_s),
    ]

    eval_set_path = Path(
        args.eval_set
        or str(config.get("evaluation_set") or "evaluation/evaluation_set_v2.json")
    )
    samples = _load_debug_samples(eval_set_path, limit=20)
    debug_results = [
        _run_debug_request(args.debug_url, sample, args.request_timeout_s)
        for sample in samples
    ]

    algo_version = str(
        os.getenv("RECOMMEND_ALGO_VERSION") or config.get("algo_version") or "v1"
    )

    lines = [
        "# Recommendation Baseline Snapshot",
        "",
        f"- Date: `{snapshot_date}`",
        "- Captured by `evaluation/capture_baseline.py`",
        f"- RECOMMEND_ALGO_VERSION: `{algo_version}`",
        f"- Evaluator config: `{config_path}`",
        f"- Eval set: `{eval_set_path}`",
        f"- Command mode: `{command_mode}`",
        f"- Eval command: `{eval_command}`",
        f"- Metrics command: `{metrics_command}`",
        "",
        "## Commands",
        "",
        _render_command_section(command_results[0]),
        "",
        _render_command_section(command_results[1]),
        "",
        _render_debug_section(debug_results),
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote baseline snapshot to {output_path}")


if __name__ == "__main__":
    main()
