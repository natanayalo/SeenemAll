"""Offline evaluation runner for recommendation quality."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover - optional local-eval path
    TestClient = None

RecItem = Tuple[int, str | None]
RecKey = Tuple[int, str | None]
GoldenTarget = Tuple[int, str | None, float]
ParamSpec = Dict[str, Any]

DEFAULT_CONFIG_PATH = "evaluation/evaluation_config.json"
DEFAULT_JUDGE_PROVIDER = "offline_golden_set"
LOCAL_BASE_PREFIX = "local://"
_LOCAL_CLIENT: Any = None


def _local_client() -> Any:
    global _LOCAL_CLIENT
    if _LOCAL_CLIENT is not None:
        return _LOCAL_CLIENT
    if TestClient is None:
        raise RuntimeError(
            "Local evaluator mode requires fastapi[testclient] dependencies."
        )
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from api.main import app

    _LOCAL_CLIENT = TestClient(app)
    return _LOCAL_CLIENT


def _normalize_local_path(base_url: str) -> str:
    path = base_url[len(LOCAL_BASE_PREFIX) :].strip()
    if not path:
        path = "/recommend"
    if not path.startswith("/"):
        path = f"/{path}"
    return path


def load_evaluation_config(path: str) -> Dict[str, Any]:
    """Load evaluator configuration from JSON (optional)."""
    config_path = Path(path)
    if not path or not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Evaluation config root must be a JSON object.")
    return loaded


def _choose_value(cli_value: Any, config_value: Any, fallback: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return fallback


def _default_param_specs() -> List[ParamSpec]:
    return [
        {"name": "default", "mode": "static", "params": {"use_llm_intent": True}},
        {
            "name": "ann_only",
            "mode": "static",
            "params": {
                "mixer_ann_weight": 1.2,
                "mixer_collab_weight": 0.0,
                "mixer_trending_weight": 0.0,
                "mixer_popularity_weight": 0.0,
                "mixer_vote_weight": 0.0,
                "mixer_novelty_weight": 0.0,
                "diversify": False,
                "use_llm_intent": False,
            },
        },
        {
            "name": "collab_boost",
            "mode": "static",
            "params": {
                "mixer_ann_weight": 0.4,
                "mixer_collab_weight": 0.8,
                "mixer_trending_weight": 0.2,
                "use_llm_intent": False,
            },
        },
        {
            "name": "popularity_boost",
            "mode": "static",
            "params": {
                "mixer_ann_weight": 0.3,
                "mixer_collab_weight": 0.2,
                "mixer_trending_weight": 0.2,
                "mixer_popularity_weight": 1.0,
                "mixer_vote_weight": 0.6,
                "use_llm_intent": False,
            },
        },
        {
            "name": "no_diversify",
            "mode": "static",
            "params": {"diversify": False, "use_llm_intent": False},
        },
        {
            "name": "genre_override",
            "mode": "genre_override_if_present",
            "params": {"use_llm_intent": False},
        },
    ]


def _load_param_specs(config: Dict[str, Any]) -> List[ParamSpec]:
    raw_specs = config.get("param_grid")
    if raw_specs is None:
        return _default_param_specs()
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("Config key 'param_grid' must be a non-empty list.")

    parsed_specs: List[ParamSpec] = []
    for index, raw in enumerate(raw_specs):
        if not isinstance(raw, dict):
            raise ValueError(f"param_grid[{index}] must be an object.")
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError(f"param_grid[{index}] is missing non-empty 'name'.")
        mode = str(raw.get("mode", "static")).strip() or "static"
        if mode not in {"static", "genre_override_if_present"}:
            raise ValueError(
                f"param_grid[{index}] mode '{mode}' is not supported. "
                "Use 'static' or 'genre_override_if_present'."
            )
        params = raw.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError(f"param_grid[{index}] field 'params' must be an object.")
        parsed_specs.append({"name": name, "mode": mode, "params": dict(params)})
    return parsed_specs


def _params_for_entry(spec: ParamSpec, entry: Dict[str, Any]) -> Dict[str, Any] | None:
    mode = str(spec.get("mode", "static"))
    params = dict(spec.get("params") or {})
    if mode == "static":
        return params
    if mode == "genre_override_if_present":
        genre_override = entry.get("genre_override")
        if not genre_override:
            return None
        params["genre_override"] = genre_override
        return params
    raise ValueError(f"Unsupported param-grid mode: {mode}")


def _resolve_judge_provider(
    *,
    cli_value: str | None,
    config: Dict[str, Any],
) -> str:
    judge_config = config.get("judge", {}) if isinstance(config.get("judge"), dict) else {}
    provider = str(
        _choose_value(
            cli_value,
            judge_config.get("provider"),
            DEFAULT_JUDGE_PROVIDER,
        )
    ).strip()
    if provider != DEFAULT_JUDGE_PROVIDER:
        raise ValueError(
            "Only judge provider 'offline_golden_set' is implemented today. "
            f"Configured provider '{provider}' is reserved for future integrations."
        )
    return provider


def load_evaluation_set(path: str) -> List[Dict[str, Any]]:
    """Load the evaluation set from JSON."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_manifest(path: str) -> Dict[str, str]:
    """Load a split manifest mapping case_id -> split name."""
    manifest_path = Path(path)
    if not path or not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    raw_assignments = loaded.get("splits") if isinstance(loaded, dict) else loaded
    if not isinstance(raw_assignments, dict):
        raise ValueError("Split manifest must be a JSON object or contain a 'splits' object.")

    assignments: Dict[str, str] = {}
    for raw_case_id, raw_split in raw_assignments.items():
        case_id = str(raw_case_id).strip()
        split_name = str(raw_split).strip().lower()
        if not case_id or not split_name:
            raise ValueError("Split manifest entries must include non-empty case ids and split names.")
        assignments[case_id] = split_name
    return assignments


def _normalise_media_type(media_type: Any) -> str | None:
    normalized_media = (
        str(media_type).strip().lower() if isinstance(media_type, str) else None
    )
    if normalized_media not in {"movie", "tv"}:
        return None
    return normalized_media


def _safe_relevance(raw_value: Any, default: float = 1.0) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(3.0, value))


def normalise_golden_targets(entry: Dict[str, Any]) -> List[GoldenTarget]:
    """Return the list of golden targets as (tmdb_id, media_type, relevance)."""
    targets: Dict[RecKey, float] = {}
    if "golden_set" not in entry:
        raise KeyError("Entry must include 'golden_set'.")
    for row in entry["golden_set"]:
        if "id" not in row:
            raise ValueError("golden_set item is missing required field 'id'.")
        media_type = _normalise_media_type(row.get("media_type"))
        if media_type is None:
            raise ValueError(
                f"golden_set item id={row['id']} must include media_type in {{movie,tv}}."
            )
        target_key: RecKey = (int(row["id"]), media_type)
        relevance = _safe_relevance(row.get("relevance"), default=1.0)
        if relevance <= 0:
            continue
        targets[target_key] = max(targets.get(target_key, 0.0), relevance)
    return [
        (tmdb_id, media_type, relevance)
        for (tmdb_id, media_type), relevance in targets.items()
    ]


def normalise_negative_targets(entry: Dict[str, Any]) -> List[RecKey]:
    """Return the list of negative targets as (tmdb_id, media_type)."""
    negatives: Dict[RecKey, None] = {}
    if "negative_set" in entry:
        for row in entry["negative_set"]:
            if "id" not in row:
                raise ValueError("negative_set item is missing required field 'id'.")
            media_type = _normalise_media_type(row.get("media_type"))
            if media_type is None:
                raise ValueError(
                    f"negative_set item id={row['id']} must include media_type in {{movie,tv}}."
                )
            key: RecKey = (int(row["id"]), media_type)
            negatives[key] = None
    return list(negatives.keys())


def _slice_tags(entry: Dict[str, Any]) -> List[str]:
    tags = [str(tag).strip() for tag in entry.get("slice_tags", []) if str(tag).strip()]
    if entry.get("query") in (None, ""):
        tags.append("no_query")
    else:
        tags.append("query")
    return list(dict.fromkeys(tags))


def _distribution_bucket(entry: Dict[str, Any]) -> str:
    raw = entry.get("distribution_bucket")
    if isinstance(raw, str):
        return raw.strip()
    return ""


def _entry_case_id(entry: Dict[str, Any]) -> str:
    raw = entry.get("case_id")
    if isinstance(raw, str):
        return raw.strip()
    if raw is None:
        return ""
    return str(raw).strip()


def _matches_any_filter(values: List[str], allowed: set[str]) -> bool:
    return not allowed or any(value in allowed for value in values)


def filter_evaluation_entries(
    evaluation_set: List[Dict[str, Any]],
    *,
    slice_tags: List[str] | None = None,
    distribution_buckets: List[str] | None = None,
    case_ids: List[str] | None = None,
    split_assignments: Dict[str, str] | None = None,
    splits: List[str] | None = None,
) -> List[Dict[str, Any]]:
    allowed_tags = {tag.strip() for tag in slice_tags or [] if tag.strip()}
    allowed_buckets = {
        bucket.strip() for bucket in distribution_buckets or [] if bucket.strip()
    }
    allowed_case_ids = {case_id.strip() for case_id in case_ids or [] if case_id.strip()}
    allowed_splits = {split.strip().lower() for split in splits or [] if split.strip()}
    if not allowed_tags and not allowed_buckets and not allowed_case_ids and not allowed_splits:
        return evaluation_set

    filtered: List[Dict[str, Any]] = []
    for entry in evaluation_set:
        tags = _slice_tags(entry)
        bucket = _distribution_bucket(entry)
        case_id = _entry_case_id(entry)
        split_name = ""
        if split_assignments is not None:
            split_name = split_assignments.get(case_id, "").strip().lower()
        if not _matches_any_filter(tags, allowed_tags):
            continue
        if allowed_buckets and bucket not in allowed_buckets:
            continue
        if allowed_case_ids and case_id not in allowed_case_ids:
            continue
        if allowed_splits and split_name not in allowed_splits:
            continue
        filtered.append(entry)
    return filtered


def _make_eval_case_id(entry: Dict[str, Any], index: int) -> str:
    provided = entry.get("case_id")
    if isinstance(provided, str) and provided.strip():
        return provided.strip()
    return f"case_{index:03d}"


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    minutes, secs = divmod(total_seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%SZ")


def _build_progress_line(
    *,
    completed_requests: int,
    total_requests: int,
    elapsed_s: float,
    last_latency_ms: float,
    params_name: str | None = None,
    params_completed: int | None = None,
    params_total: int | None = None,
) -> str:
    progress_pct = 100.0 if total_requests <= 0 else (completed_requests / total_requests) * 100.0
    average_seconds = 0.0 if completed_requests <= 0 else elapsed_s / completed_requests
    remaining_requests = max(total_requests - completed_requests, 0)
    eta_s = average_seconds * remaining_requests
    prefix = "Progress"
    if params_name and params_completed is not None and params_total is not None:
        prefix = f"{params_name} {params_completed}/{params_total} | total"
    return (
        f"[{_utc_timestamp()}] {prefix} {completed_requests}/{total_requests} "
        f"({progress_pct:.1f}%) | elapsed {_format_duration(elapsed_s)} "
        f"| eta {_format_duration(eta_s)} | last {last_latency_ms:.0f}ms"
    )


def _should_emit_progress(
    *,
    completed_requests: int,
    total_requests: int,
    progress_every: int,
    log_level: str,
) -> bool:
    if log_level == "verbose":
        return True
    interval = max(progress_every, 1)
    return completed_requests >= total_requests or completed_requests % interval == 0


def _build_eval_plan(
    evaluation_set: List[Dict[str, Any]],
    param_specs: List[ParamSpec],
) -> List[Tuple[ParamSpec, List[Tuple[int, Dict[str, Any], Dict[str, Any]]]]]:
    plan: List[Tuple[ParamSpec, List[Tuple[int, Dict[str, Any], Dict[str, Any]]]]] = []
    for param_spec in param_specs:
        entries: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
        for index, entry in enumerate(evaluation_set):
            base_params = _params_for_entry(param_spec, entry)
            if base_params is None:
                continue
            entries.append((index, entry, base_params))
        plan.append((param_spec, entries))
    return plan


def _validate_split_assignments(
    evaluation_set: List[Dict[str, Any]],
    split_assignments: Dict[str, str],
) -> None:
    missing_case_ids: List[str] = []
    for index, entry in enumerate(evaluation_set):
        case_id = _make_eval_case_id(entry, index)
        if case_id not in split_assignments:
            missing_case_ids.append(case_id)
    if missing_case_ids:
        preview = ", ".join(missing_case_ids[:5])
        raise ValueError(
            "Split manifest is missing assignments for evaluation cases: "
            f"{preview}"
        )


def _format_compact_summary_table(summary: Dict[str, Any], k: int) -> List[str]:
    rows: List[Dict[str, str]] = []
    include_delta = False
    for params_name, payload in summary.items():
        if params_name.startswith("_"):
            continue
        overall = payload["overall"]
        delta = payload.get("delta_vs_baseline", {}).get("overall")
        if delta:
            include_delta = True
        rows.append(
            {
                "scenario": params_name,
                "count": str(int(overall["count"])),
                f"nDCG@{k}": f"{overall[f'ndcg@{k}']:.4f}",
                "MAP": f"{overall['map']:.4f}",
                f"Hit@{k}": f"{overall[f'hit_rate@{k}']:.4f}",
                "lat_ms": f"{overall['latency_ms_avg']:.2f}",
                "dNDCG": f"{delta[f'ndcg@{k}']:+.4f}" if delta else "",
                "dMAP": f"{delta['map']:+.4f}" if delta else "",
            }
        )

    headers = ["scenario", "count", f"nDCG@{k}", "MAP", f"Hit@{k}", "lat_ms"]
    if include_delta:
        headers.extend(["dNDCG", "dMAP"])
    widths = {
        header: max(len(header), *(len(row.get(header, "")) for row in rows))
        for header in headers
    }

    def _format_row(row: Dict[str, str]) -> str:
        return "  ".join(
            row.get(header, "").ljust(widths[header]) if header == "scenario" else row.get(header, "").rjust(widths[header])
            for header in headers
        )

    return [_format_row({header: header for header in headers})] + [
        _format_row(row) for row in rows
    ]


def _item_to_token(item: RecItem) -> str:
    tmdb_id, media_type = item
    return f"{tmdb_id}:{media_type or '*'}"


def _items_to_pipe(items: List[RecItem]) -> str:
    return "|".join(_item_to_token(item) for item in items)


def _target_to_token(target: GoldenTarget) -> str:
    tmdb_id, media_type, relevance = target
    return f"{tmdb_id}:{media_type or '*'}@{relevance:.2f}"


def _targets_to_pipe(targets: List[GoldenTarget]) -> str:
    return "|".join(_target_to_token(target) for target in targets)


def _pipe_to_items(raw: str | None) -> List[RecItem]:
    if not raw:
        return []
    parts = [chunk.strip() for chunk in str(raw).split("|")]
    result: List[RecItem] = []
    for part in parts:
        if not part:
            continue
        if ":" in part:
            raw_id, raw_media = part.split(":", 1)
            try:
                item_id = int(raw_id)
            except ValueError:
                continue
            media = raw_media.strip().lower()
            if media in {"", "*"}:
                media = None
            result.append((item_id, media if media in {"movie", "tv"} else None))
            continue
        try:
            result.append((int(part), None))
        except ValueError:
            continue
    return result


def _golden_lookup(
    golden_targets: List[GoldenTarget],
) -> dict[RecKey, float]:
    exact: dict[RecKey, float] = {}
    for tmdb_id, media_type, relevance in golden_targets:
        if relevance <= 0:
            continue
        key: RecKey = (tmdb_id, media_type)
        exact[key] = max(exact.get(key, 0.0), relevance)
    return exact


def _is_hit(item: RecItem, exact: dict[RecKey, float]) -> bool:
    return _item_relevance(item, exact) > 0.0


def _item_relevance(
    item: RecItem,
    exact: dict[RecKey, float],
) -> float:
    return float(exact.get(item, 0.0))


def _relevant_target_count(golden_targets: List[GoldenTarget]) -> int:
    return sum(1 for _, _, relevance in golden_targets if relevance > 0.0)


def call_recommendation_api(
    *,
    base_url: str,
    query: str | None,
    params: Dict[str, Any],
    request_overrides: Dict[str, Any] | None,
    timeout_s: float,
) -> Tuple[List[RecItem], float]:
    """Call recommendation API and return ((tmdb_id, media_type) list, latency_ms)."""
    all_params: Dict[str, Any] = dict(params)
    if request_overrides:
        all_params.update(request_overrides)
    if query not in (None, ""):
        all_params["query"] = query

    if base_url.startswith(LOCAL_BASE_PREFIX):
        path = _normalize_local_path(base_url)
        try:
            client = _local_client()
            start = time.perf_counter()
            response = client.get(path, params=all_params)
            latency_ms = (time.perf_counter() - start) * 1000.0
            response.raise_for_status()
            data = response.json()
            items: List[RecItem] = []
            for item in data.get("items", []):
                if "tmdb_id" not in item:
                    continue
                media_type = item.get("media_type")
                normalized_media = (
                    str(media_type).strip().lower() if isinstance(media_type, str) else None
                )
                if normalized_media not in {"movie", "tv"}:
                    normalized_media = None
                items.append((int(item["tmdb_id"]), normalized_media))
            return items, float(latency_ms)
        except Exception as exc:
            print(f"Local API request failed for params={all_params}: {exc}")
            return [], 0.0

    try:
        response = httpx.get(base_url, params=all_params, timeout=timeout_s)
        response.raise_for_status()
        data = response.json()
        items: List[RecItem] = []
        for item in data.get("items", []):
            if "tmdb_id" not in item:
                continue
            media_type = item.get("media_type")
            normalized_media = (
                str(media_type).strip().lower() if isinstance(media_type, str) else None
            )
            if normalized_media not in {"movie", "tv"}:
                normalized_media = None
            items.append((int(item["tmdb_id"]), normalized_media))
        return items, float(response.elapsed.total_seconds() * 1000.0)
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
        print(f"API request failed for params={all_params}: {exc}")
        return [], 0.0


def calculate_precision_at_k(
    recommended_items: List[RecItem], golden_targets: List[GoldenTarget], k: int
) -> float:
    if not recommended_items or k <= 0:
        return 0.0
    exact = _golden_lookup(golden_targets)
    hits = 0
    for rec in recommended_items[:k]:
        if _is_hit(rec, exact):
            hits += 1
    return hits / k


def calculate_recall_at_k(
    recommended_items: List[RecItem], golden_targets: List[GoldenTarget], k: int
) -> float:
    relevant_count = _relevant_target_count(golden_targets)
    if relevant_count == 0:
        return 0.0
    exact = _golden_lookup(golden_targets)
    hit_targets: set[RecKey] = set()
    for rec in recommended_items[:k]:
        if rec in exact and exact[rec] > 0:
            hit_targets.add(rec)
    return len(hit_targets) / relevant_count


def calculate_average_precision(
    recommended_items: List[RecItem], golden_targets: List[GoldenTarget]
) -> float:
    relevant_count = _relevant_target_count(golden_targets)
    if relevant_count == 0:
        return 0.0

    exact = _golden_lookup(golden_targets)
    matched: set[RecKey] = set()
    hits = 0
    precision_sum = 0.0
    for idx, rec in enumerate(recommended_items):
        relevance = _item_relevance(rec, exact)
        if relevance <= 0:
            continue
        if rec in matched:
            continue
        matched.add(rec)
        hits += 1
        precision_sum += hits / (idx + 1)
    return precision_sum / relevant_count


def calculate_ndcg_at_k(
    recommended_items: List[RecItem], golden_targets: List[GoldenTarget], k: int
) -> float:
    exact = _golden_lookup(golden_targets)
    matched: set[RecKey] = set()
    dcg = 0.0
    for idx, rec in enumerate(recommended_items[:k]):
        relevance = _item_relevance(rec, exact)
        if relevance <= 0:
            continue
        if rec in matched:
            continue
        matched.add(rec)
        gain = (2**relevance) - 1.0
        dcg += gain / math.log2(idx + 2)

    relevant_relevances = sorted(
        [float(relevance) for _, _, relevance in golden_targets if relevance > 0.0],
        reverse=True,
    )
    idcg = 0.0
    for idx, relevance in enumerate(relevant_relevances[:k]):
        gain = (2**relevance) - 1.0
        idcg += gain / math.log2(idx + 2)

    return dcg / idcg if idcg > 0 else 0.0


def calculate_hit_rate_at_k(
    recommended_items: List[RecItem], golden_targets: List[GoldenTarget], k: int
) -> float:
    if not recommended_items or not golden_targets or k <= 0:
        return 0.0
    exact = _golden_lookup(golden_targets)
    return 1.0 if any(_is_hit(rec, exact) for rec in recommended_items[:k]) else 0.0


def calculate_mrr_at_k(
    recommended_items: List[RecItem], golden_targets: List[GoldenTarget], k: int
) -> float:
    if not recommended_items or not golden_targets or k <= 0:
        return 0.0
    exact = _golden_lookup(golden_targets)
    for idx, rec in enumerate(recommended_items[:k]):
        if _item_relevance(rec, exact) > 0:
            return 1.0 / float(idx + 1)
    return 0.0


def _negative_lookup(negative_targets: List[RecKey]) -> set[RecKey]:
    return set(negative_targets)


def calculate_negative_rate_at_k(
    recommended_items: List[RecItem], negative_targets: List[RecKey], k: int
) -> float:
    if k <= 0 or not recommended_items or not negative_targets:
        return 0.0
    negatives_lookup = _negative_lookup(negative_targets)
    negatives = 0
    for rec in recommended_items[:k]:
        if rec in negatives_lookup:
            negatives += 1
    return negatives / k


def calculate_negative_hit_rate_at_k(
    recommended_items: List[RecItem], negative_targets: List[RecKey], k: int
) -> float:
    if k <= 0 or not recommended_items or not negative_targets:
        return 0.0
    negatives_lookup = _negative_lookup(negative_targets)
    return 1.0 if any(rec in negatives_lookup for rec in recommended_items[:k]) else 0.0


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    low_value = ordered[lower]
    high_value = ordered[upper]
    weight = rank - lower
    return float(low_value + (high_value - low_value) * weight)


def _bootstrap_mean_ci(
    values: List[float],
    *,
    samples: int,
    seed: int,
) -> Dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0}
    if len(values) == 1:
        singleton = float(values[0])
        return {"low": singleton, "high": singleton}

    rng = random.Random(seed)
    n = len(values)
    sample_means: List[float] = []
    for _ in range(samples):
        sampled = [values[rng.randrange(n)] for _ in range(n)]
        sample_means.append(_avg(sampled))
    return {
        "low": round(_percentile(sample_means, 0.025), 4),
        "high": round(_percentile(sample_means, 0.975), 4),
    }


def aggregate_metrics(
    rows: List[Dict[str, Any]],
    k: int,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    ci_label: str,
) -> Dict[str, Any]:
    precision_key = f"precision@{k}"
    recall_key = f"recall@{k}"
    ndcg_key = f"ndcg@{k}"
    hit_rate_key = f"hit_rate@{k}"
    mrr_key = f"mrr@{k}"
    negative_rate_key = f"negative_rate@{k}"
    negative_hit_rate_key = f"negative_hit_rate@{k}"

    metric_values: Dict[str, List[float]] = {
        precision_key: [float(row[precision_key]) for row in rows],
        recall_key: [float(row[recall_key]) for row in rows],
        "map": [float(row["average_precision"]) for row in rows],
        ndcg_key: [float(row[ndcg_key]) for row in rows],
        hit_rate_key: [float(row[hit_rate_key]) for row in rows],
        mrr_key: [float(row[mrr_key]) for row in rows],
        negative_rate_key: [float(row[negative_rate_key]) for row in rows],
        negative_hit_rate_key: [float(row[negative_hit_rate_key]) for row in rows],
        "latency_ms_avg": [float(row["latency_ms"]) for row in rows],
    }

    metrics: Dict[str, Any] = {
        "count": float(len(rows)),
        precision_key: _avg(metric_values[precision_key]),
        recall_key: _avg(metric_values[recall_key]),
        "map": _avg(metric_values["map"]),
        ndcg_key: _avg(metric_values[ndcg_key]),
        hit_rate_key: _avg(metric_values[hit_rate_key]),
        mrr_key: _avg(metric_values[mrr_key]),
        negative_rate_key: _avg(metric_values[negative_rate_key]),
        negative_hit_rate_key: _avg(metric_values[negative_hit_rate_key]),
        "latency_ms_avg": _avg(metric_values["latency_ms_avg"]),
    }

    if bootstrap_samples > 0 and rows:
        for metric_name in (
            precision_key,
            recall_key,
            "map",
            ndcg_key,
            hit_rate_key,
            mrr_key,
            negative_rate_key,
            negative_hit_rate_key,
            "latency_ms_avg",
        ):
            seed = bootstrap_seed + abs(hash(f"{ci_label}:{metric_name}")) % 100000
            metrics[f"{metric_name}_ci95"] = _bootstrap_mean_ci(
                metric_values[metric_name],
                samples=bootstrap_samples,
                seed=seed,
            )

    return metrics


def _delta_dict(
    metrics: Dict[str, Any],
    baseline: Dict[str, Any],
    *,
    keys: List[str],
) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for key in keys:
        deltas[key] = round(float(metrics.get(key, 0.0) - baseline.get(key, 0.0)), 4)
    return deltas


def _compare_row_key(row: Dict[str, Any]) -> Tuple[str, str]:
    params_name = str(row.get("params_name", ""))
    case_id = str(row.get("eval_case_id", ""))
    return (params_name, case_id)


def _load_compare_rows(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        loaded = list(reader)

    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in loaded:
        if not row.get("eval_case_id"):
            raise ValueError(
                "Compare CSV is missing required 'eval_case_id'. "
                "Re-run baseline with the current evaluator."
            )
        key = _compare_row_key(row)
        by_key[key] = row
    return by_key


def _per_query_diffs(
    rows: List[Dict[str, Any]],
    baseline_rows: Dict[Tuple[str, str], Dict[str, Any]],
    *,
    k: int,
) -> List[Dict[str, Any]]:
    precision_key = f"precision@{k}"
    recall_key = f"recall@{k}"
    ndcg_key = f"ndcg@{k}"
    hit_key = f"hit_rate@{k}"
    mrr_key = f"mrr@{k}"
    negative_rate_key = f"negative_rate@{k}"
    negative_hit_rate_key = f"negative_hit_rate@{k}"

    diffs: List[Dict[str, Any]] = []
    for row in rows:
        key = _compare_row_key(row)
        baseline_row = baseline_rows.get(key)
        if baseline_row is None:
            continue

        cand_hit_items = _pipe_to_items(str(row.get("hit_items_top_k", "")))
        base_hit_items = _pipe_to_items(str(baseline_row.get("hit_items_top_k", "")))

        cand_rec_items = _pipe_to_items(str(row.get("recommended_items_top_k", "")))
        base_rec_items = _pipe_to_items(str(baseline_row.get("recommended_items_top_k", "")))

        cand_hits = {_item_to_token(item) for item in cand_hit_items}
        base_hits = {_item_to_token(item) for item in base_hit_items}
        cand_recs = {_item_to_token(item) for item in cand_rec_items}
        base_recs = {_item_to_token(item) for item in base_rec_items}

        ndcg_delta = round(float(row[ndcg_key]) - float(baseline_row.get(ndcg_key, 0.0)), 4)
        status = "no_change"
        if ndcg_delta > 0:
            status = "improved"
        if ndcg_delta < 0:
            status = "regressed"

        diffs.append(
            {
                "params_name": row["params_name"],
                "eval_case_id": row["eval_case_id"],
                "query": row["query"],
                "slice_tags": row["slice_tags"],
                "status": status,
                f"{precision_key}_delta": round(
                    float(row[precision_key]) - float(baseline_row.get(precision_key, 0.0)),
                    4,
                ),
                f"{recall_key}_delta": round(
                    float(row[recall_key]) - float(baseline_row.get(recall_key, 0.0)),
                    4,
                ),
                "average_precision_delta": round(
                    float(row["average_precision"])
                    - float(baseline_row.get("average_precision", 0.0)),
                    4,
                ),
                f"{ndcg_key}_delta": ndcg_delta,
                f"{hit_key}_delta": round(
                    float(row[hit_key]) - float(baseline_row.get(hit_key, 0.0)),
                    4,
                ),
                f"{mrr_key}_delta": round(
                    float(row.get(mrr_key, 0.0)) - float(baseline_row.get(mrr_key, 0.0)),
                    4,
                ),
                f"{negative_rate_key}_delta": round(
                    float(row.get(negative_rate_key, 0.0))
                    - float(baseline_row.get(negative_rate_key, 0.0)),
                    4,
                ),
                f"{negative_hit_rate_key}_delta": round(
                    float(row.get(negative_hit_rate_key, 0.0))
                    - float(baseline_row.get(negative_hit_rate_key, 0.0)),
                    4,
                ),
                "latency_ms_delta": round(
                    float(row["latency_ms"]) - float(baseline_row.get("latency_ms", 0.0)),
                    2,
                ),
                "added_hits": sorted(cand_hits - base_hits),
                "lost_hits": sorted(base_hits - cand_hits),
                "added_recommended_items": sorted(cand_recs - base_recs),
                "lost_recommended_items": sorted(base_recs - cand_recs),
                "candidate_recommended_items_top_k": [
                    _item_to_token(item) for item in cand_rec_items
                ],
                "baseline_recommended_items_top_k": [
                    _item_to_token(item) for item in base_rec_items
                ],
            }
        )

    diffs.sort(key=lambda item: float(item[f"{ndcg_key}_delta"]))
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline recommendation evaluation.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Optional evaluator config JSON. "
            "If file is missing, built-in defaults are used."
        ),
    )
    parser.add_argument(
        "--evaluation-set",
        default=None,
        help="Path to evaluation set JSON.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Recommendation endpoint URL.",
    )
    parser.add_argument("--k", type=int, default=None, help="Top-K cut for metrics.")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="HTTP timeout in seconds per recommendation call.",
    )
    parser.add_argument(
        "--default-user-id",
        default=None,
        help="Default user_id used when an entry does not override it.",
    )
    parser.add_argument(
        "--baseline-name",
        default=None,
        help="Parameter-set name used as baseline for deltas.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="Bootstrap samples for 95%% confidence intervals (0 disables CIs).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=None,
        help="Seed for bootstrap CI generation.",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        help=(
            "Judge provider id. "
            "Current implementation supports only 'offline_golden_set'."
        ),
    )
    parser.add_argument(
        "--compare-csv",
        default="",
        help="Optional baseline detailed CSV for per-query diff output.",
    )
    parser.add_argument(
        "--per-query-diff-json",
        default="",
        help="Optional per-query diff JSON output path (requires --compare-csv).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Detailed per-query result CSV output path.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Summary JSON output path.",
    )
    parser.add_argument(
        "--slice-tag",
        action="append",
        default=[],
        help=(
            "Restrict evaluation cases to entries containing this slice tag. "
            "Repeat the flag to allow multiple tags."
        ),
    )
    parser.add_argument(
        "--distribution-bucket",
        action="append",
        default=[],
        help=(
            "Restrict evaluation cases to these distribution buckets. "
            "Repeat the flag to allow multiple buckets."
        ),
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help=(
            "Restrict evaluation to specific case ids. "
            "Repeat the flag to target multiple cases."
        ),
    )
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Optional JSON manifest mapping case ids to stable split names.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        help=(
            "Restrict evaluation to these split names from --split-manifest. "
            "Repeat the flag to allow multiple splits."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["compact", "verbose"],
        default="compact",
        help=(
            "compact prints throttled progress plus a one-line summary table. "
            "verbose prints per-case metrics."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Emit one compact progress update every N completed requests.",
    )
    parser.add_argument(
        "--skip-output-csv",
        action="store_true",
        help="Skip writing the detailed per-query CSV and keep only the summary JSON.",
    )
    args = parser.parse_args()

    config = load_evaluation_config(args.config)
    config_api = config.get("api", {}) if isinstance(config.get("api"), dict) else {}
    config_metrics = (
        config.get("metrics", {}) if isinstance(config.get("metrics"), dict) else {}
    )
    config_paths = config.get("paths", {}) if isinstance(config.get("paths"), dict) else {}

    evaluation_set_path = str(
        _choose_value(
            args.evaluation_set,
            config.get("evaluation_set"),
            "evaluation/evaluation_set.json",
        )
    )
    base_url = str(
        _choose_value(
            args.base_url,
            config_api.get("base_url"),
            "http://localhost:8000/recommend",
        )
    )
    timeout_s = float(_choose_value(args.timeout_s, config_api.get("timeout_s"), 8.0))
    default_user_id = str(
        _choose_value(args.default_user_id, config_api.get("default_user_id"), "u1")
    )
    baseline_name = str(
        _choose_value(args.baseline_name, config_metrics.get("baseline_name"), "default")
    )
    k = max(int(_choose_value(args.k, config_metrics.get("k"), 10)), 1)
    bootstrap_samples = int(
        _choose_value(args.bootstrap_samples, config_metrics.get("bootstrap_samples"), 400)
    )
    bootstrap_seed = int(
        _choose_value(args.bootstrap_seed, config_metrics.get("bootstrap_seed"), 42)
    )
    output_csv_path = str(
        _choose_value(
            args.output_csv,
            config_paths.get("output_csv"),
            "evaluation/artifacts/evaluation_results_latest.csv",
        )
    )
    summary_json_path = str(
        _choose_value(
            args.summary_json,
            config_paths.get("summary_json"),
            "evaluation/artifacts/evaluation_summary_latest.json",
        )
    )
    judge_provider = _resolve_judge_provider(cli_value=args.judge_provider, config=config)
    algo_version = str(config.get("algo_version", "v1"))
    config_splits = config.get("splits", {}) if isinstance(config.get("splits"), dict) else {}
    split_manifest_path = _choose_value(
        args.split_manifest,
        config_splits.get("manifest"),
        None,
    )

    raw_evaluation_set = load_evaluation_set(evaluation_set_path)
    split_assignments: Dict[str, str] | None = None
    if split_manifest_path:
        split_assignments = load_split_manifest(str(split_manifest_path))
        _validate_split_assignments(raw_evaluation_set, split_assignments)

    evaluation_set = filter_evaluation_entries(
        raw_evaluation_set,
        slice_tags=args.slice_tag,
        distribution_buckets=args.distribution_bucket,
        case_ids=args.case_id,
        split_assignments=split_assignments,
        splits=args.split,
    )
    param_specs = _load_param_specs(config)
    eval_plan = _build_eval_plan(evaluation_set, param_specs)
    total_requests = sum(len(entries) for _, entries in eval_plan)
    if total_requests <= 0:
        raise ValueError("No evaluation cases remain after applying the requested filters.")

    print(
        f"Starting evaluation: {len(evaluation_set)} cases x {len(param_specs)} parameter sets "
        f"= {total_requests} requests"
    )
    if args.slice_tag:
        print(f"  slice tag filter: {', '.join(args.slice_tag)}")
    if args.distribution_bucket:
        print(f"  distribution bucket filter: {', '.join(args.distribution_bucket)}")
    if args.case_id:
        print(f"  case id filter: {', '.join(args.case_id)}")
    if args.split:
        print(f"  split filter: {', '.join(args.split)}")
    if split_manifest_path:
        print(f"  split manifest: {split_manifest_path}")
    print(f"  base_url: {base_url}")

    all_rows: List[Dict[str, Any]] = []
    scenario_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    started_at = time.perf_counter()
    completed_requests = 0

    for param_index, (param_spec, planned_entries) in enumerate(eval_plan, start=1):
        params_name = str(param_spec["name"])
        if args.log_level == "verbose":
            print(f"--- Evaluating parameters: {params_name} ---")
        else:
            print(
                f"[{_utc_timestamp()}] Evaluating {params_name} "
                f"({param_index}/{len(eval_plan)}) | requests {len(planned_entries)}"
            )

        for params_completed, (index, entry, base_params) in enumerate(planned_entries, start=1):
            query_value = entry.get("query")
            query: str | None = None
            if query_value not in (None, ""):
                query = str(query_value)
            display_query = query if query else "<NO_QUERY>"

            try:
                golden_targets = normalise_golden_targets(entry)
                negative_targets = normalise_negative_targets(entry)
            except (KeyError, ValueError) as exc:
                print(f"Skipping entry '{display_query}': {exc}")
                continue

            request_overrides = dict(entry.get("request_overrides") or {})
            if "user_id" not in request_overrides and default_user_id:
                request_overrides["user_id"] = default_user_id

            recommended_items, latency_ms = call_recommendation_api(
                base_url=base_url,
                query=query,
                params=base_params,
                request_overrides=request_overrides,
                timeout_s=timeout_s,
            )

            precision = calculate_precision_at_k(recommended_items, golden_targets, k)
            recall = calculate_recall_at_k(recommended_items, golden_targets, k)
            ap = calculate_average_precision(recommended_items, golden_targets)
            ndcg = calculate_ndcg_at_k(recommended_items, golden_targets, k)
            hit_rate = calculate_hit_rate_at_k(recommended_items, golden_targets, k)
            mrr = calculate_mrr_at_k(recommended_items, golden_targets, k)
            negative_rate = calculate_negative_rate_at_k(recommended_items, negative_targets, k)
            negative_hit_rate = calculate_negative_hit_rate_at_k(
                recommended_items, negative_targets, k
            )
            slices = _slice_tags(entry)
            top_k_recs = recommended_items[:k]
            golden_lookup = _golden_lookup(golden_targets)
            top_k_hits = [
                rec for rec in top_k_recs if _is_hit(rec, golden_lookup)
            ]

            row = {
                "params_name": params_name,
                "eval_case_id": _make_eval_case_id(entry, index),
                "split": (
                    split_assignments.get(_make_eval_case_id(entry, index), "")
                    if split_assignments is not None
                    else ""
                ),
                "distribution_bucket": _distribution_bucket(entry),
                "query": display_query,
                "slice_tags": "|".join(slices),
                "golden_count": len(golden_targets),
                "negative_count": len(negative_targets),
                "latency_ms": round(latency_ms, 2),
                "recommended_items_top_k": _items_to_pipe(top_k_recs),
                "golden_items_with_relevance": _targets_to_pipe(golden_targets),
                "hit_items_top_k": _items_to_pipe(top_k_hits),
                "negative_items": _items_to_pipe(negative_targets),
                f"precision@{k}": round(precision, 4),
                f"recall@{k}": round(recall, 4),
                "average_precision": round(ap, 4),
                f"ndcg@{k}": round(ndcg, 4),
                f"hit_rate@{k}": round(hit_rate, 4),
                f"mrr@{k}": round(mrr, 4),
                f"negative_rate@{k}": round(negative_rate, 4),
                f"negative_hit_rate@{k}": round(negative_hit_rate, 4),
            }
            all_rows.append(row)
            scenario_to_rows[params_name].append(row)
            completed_requests += 1

            if args.log_level == "verbose":
                print(f"  Query: {display_query}")
                print(f"    case_id: {row['eval_case_id']}")
                print(f"    slices: {slices}")
                print(f"    Precision@{k}: {precision:.4f}")
                print(f"    Recall@{k}: {recall:.4f}")
                print(f"    Average Precision: {ap:.4f}")
                print(f"    nDCG@{k}: {ndcg:.4f}")
                print(f"    HitRate@{k}: {hit_rate:.4f}")
                print(f"    MRR@{k}: {mrr:.4f}")
                print(f"    NegativeRate@{k}: {negative_rate:.4f}")

            if _should_emit_progress(
                completed_requests=completed_requests,
                total_requests=total_requests,
                progress_every=args.progress_every,
                log_level=args.log_level,
            ):
                print(
                    _build_progress_line(
                        completed_requests=completed_requests,
                        total_requests=total_requests,
                        elapsed_s=time.perf_counter() - started_at,
                        last_latency_ms=latency_ms,
                        params_name=params_name,
                        params_completed=params_completed,
                        params_total=len(planned_entries),
                    ),
                    flush=True,
                )

    summary: Dict[str, Any] = {
        "_meta": {
            "k": k,
            "bootstrap_samples": max(bootstrap_samples, 0),
            "bootstrap_seed": int(bootstrap_seed),
            "baseline_name": baseline_name,
            "judge_provider": judge_provider,
            "algo_version": algo_version,
            "config_path": args.config,
            "split_manifest": split_manifest_path or "",
            "split_filter": [split.strip().lower() for split in args.split if split.strip()],
        }
    }
    tracked_metric_keys = [
        f"precision@{k}",
        f"recall@{k}",
        "map",
        f"ndcg@{k}",
        f"hit_rate@{k}",
        f"mrr@{k}",
        f"negative_rate@{k}",
        f"negative_hit_rate@{k}",
        "latency_ms_avg",
    ]

    for params_name, rows in scenario_to_rows.items():
        overall = aggregate_metrics(
            rows,
            k,
            bootstrap_samples=max(bootstrap_samples, 0),
            bootstrap_seed=int(bootstrap_seed),
            ci_label=f"{params_name}:overall",
        )
        by_slice: Dict[str, Dict[str, Any]] = {}
        unique_slices = sorted(
            {
                slice_name
                for row in rows
                for slice_name in str(row.get("slice_tags", "")).split("|")
                if slice_name
            }
        )
        for slice_name in unique_slices:
            slice_rows = [
                row
                for row in rows
                if slice_name in str(row.get("slice_tags", "")).split("|")
            ]
            by_slice[slice_name] = aggregate_metrics(
                slice_rows,
                k,
                bootstrap_samples=max(bootstrap_samples, 0),
                bootstrap_seed=int(bootstrap_seed),
                ci_label=f"{params_name}:{slice_name}",
            )

        summary[params_name] = {"overall": overall, "by_slice": by_slice}

    baseline = summary.get(baseline_name)
    if baseline:
        baseline_overall = baseline["overall"]
        baseline_by_slice = baseline["by_slice"]
        for params_name, payload in summary.items():
            if params_name.startswith("_") or params_name == baseline_name:
                continue

            payload["delta_vs_baseline"] = {
                "overall": _delta_dict(
                    payload["overall"],
                    baseline_overall,
                    keys=tracked_metric_keys,
                ),
                "by_slice": {},
            }
            for slice_name, slice_metrics in payload["by_slice"].items():
                if slice_name not in baseline_by_slice:
                    continue
                payload["delta_vs_baseline"]["by_slice"][slice_name] = _delta_dict(
                    slice_metrics,
                    baseline_by_slice[slice_name],
                    keys=tracked_metric_keys,
                )

    print("\n=== Evaluation Summary ===")
    if args.log_level == "compact":
        for line in _format_compact_summary_table(summary, k):
            print(line)
    else:
        for params_name, payload in summary.items():
            if params_name.startswith("_"):
                continue
            overall = payload["overall"]
            print(f"\n{params_name}")
            print(f"  Count: {int(overall['count'])}")
            print(f"  Precision@{k}: {overall[f'precision@{k}']:.4f}")
            print(f"  Recall@{k}: {overall[f'recall@{k}']:.4f}")
            print(f"  MAP: {overall['map']:.4f}")
            print(f"  nDCG@{k}: {overall[f'ndcg@{k}']:.4f}")
            print(f"  HitRate@{k}: {overall[f'hit_rate@{k}']:.4f}")
            print(f"  MRR@{k}: {overall[f'mrr@{k}']:.4f}")
            print(f"  NegativeRate@{k}: {overall[f'negative_rate@{k}']:.4f}")
            print(f"  Avg latency(ms): {overall['latency_ms_avg']:.2f}")

            ci_key = f"ndcg@{k}_ci95"
            ndcg_ci = overall.get(ci_key)
            if isinstance(ndcg_ci, dict):
                print(
                    f"  nDCG@{k} 95% CI: [{ndcg_ci.get('low'):.4f}, {ndcg_ci.get('high'):.4f}]"
                )

            delta = payload.get("delta_vs_baseline", {}).get("overall")
            if delta:
                print("  Delta vs baseline:")
                print(f"    Precision@{k}: {delta[f'precision@{k}']:+.4f}")
                print(f"    Recall@{k}: {delta[f'recall@{k}']:+.4f}")
                print(f"    MAP: {delta['map']:+.4f}")
                print(f"    nDCG@{k}: {delta[f'ndcg@{k}']:+.4f}")
                print(f"    HitRate@{k}: {delta[f'hit_rate@{k}']:+.4f}")
                print(f"    MRR@{k}: {delta[f'mrr@{k}']:+.4f}")
                print(f"    NegativeRate@{k}: {delta[f'negative_rate@{k}']:+.4f}")
                print(f"    Latency(ms): {delta['latency_ms_avg']:+.2f}")

    output_csv = Path(output_csv_path)
    if not args.skip_output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        if all_rows:
            with open(output_csv, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)
        else:
            print("\nNo evaluation results to save.")

    per_query_diff_output = ""
    if args.compare_csv and not args.skip_output_csv:
        compare_lookup = _load_compare_rows(Path(args.compare_csv))
        diffs = _per_query_diffs(all_rows, compare_lookup, k=k)
        if diffs:
            if args.per_query_diff_json:
                per_query_diff_path = Path(args.per_query_diff_json)
            else:
                per_query_diff_path = output_csv.with_name(
                    f"{output_csv.stem}_per_query_diff.json"
                )
            per_query_diff_path.parent.mkdir(parents=True, exist_ok=True)
            with open(per_query_diff_path, "w", encoding="utf-8") as handle:
                json.dump(diffs, handle, indent=2)
            per_query_diff_output = str(per_query_diff_path)

            if args.log_level == "verbose":
                print(f"Per-query diffs saved to {per_query_diff_path}")
                print("Top 5 regressions by nDCG delta:")
                for item in diffs[:5]:
                    print(
                        "  "
                        f"{item['params_name']} {item['eval_case_id']} "
                        f"nDCG delta={item[f'ndcg@{k}_delta']:+.4f} "
                        f"lost_hits={item['lost_hits']}"
                    )
        else:
            if args.log_level == "verbose":
                print("No per-query overlaps found for compare CSV.")

    if per_query_diff_output:
        summary["_meta"]["per_query_diff_json"] = per_query_diff_output

    summary_json = Path(summary_json_path)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    artifact_labels = [f"summary={summary_json}"]
    if not args.skip_output_csv and all_rows:
        artifact_labels.append(f"results={output_csv}")
    if per_query_diff_output:
        artifact_labels.append(f"diff={per_query_diff_output}")
    print(f"Artifacts: {' | '.join(artifact_labels)}")


if __name__ == "__main__":
    main()
