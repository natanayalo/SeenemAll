from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple, overload

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder  # pragma: no cover
else:
    CrossEncoder = Any  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DEFAULT_ALPHA = 0.7
_DEFAULT_MAX_CANDIDATES = 25
_DEFAULT_BATCH_SIZE = 32

_model_cache: Dict[str, CrossEncoder] = {}
_model_lock = threading.Lock()


class OpenVINOCrossEncoder:
    """OpenVINO-accelerated Cross-Encoder for Intel Arc GPU, NPU, and CPU."""

    def __init__(self, model_name: str, device: str = "GPU"):
        from pathlib import Path
        from optimum.intel.openvino import OVModelForSequenceClassification
        from transformers import AutoTokenizer

        self.device = device.upper()
        safe_name = model_name.replace("/", "--")
        cache_dir = Path.home() / ".cache" / "openvino_models" / safe_name

        if (cache_dir / "openvino_model.xml").exists():
            self.tokenizer = AutoTokenizer.from_pretrained(cache_dir)
            if self.device == "NPU":
                self.model = OVModelForSequenceClassification.from_pretrained(
                    cache_dir, compile=False
                )
                self.model.reshape(25, 128)
                self.model.to("NPU")
                self.model.compile()
            else:
                self.model = OVModelForSequenceClassification.from_pretrained(
                    cache_dir, device=self.device
                )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.device == "NPU":
                self.model = OVModelForSequenceClassification.from_pretrained(
                    model_name, export=True, compile=False
                )
                self.model.reshape(25, 128)
                self.model.to("NPU")
                self.model.compile()
            else:
                self.model = OVModelForSequenceClassification.from_pretrained(
                    model_name, export=True, device=self.device
                )
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(cache_dir)
                self.tokenizer.save_pretrained(cache_dir)
            except Exception as exc:
                logger.debug("Could not cache OpenVINO model to %s: %s", cache_dir, exc)

    def predict(
        self,
        sentences: Sequence[Tuple[str, str]],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        if not sentences:
            return np.array([], dtype=np.float32)

        import contextlib

        try:
            import torch

            no_grad_ctx = getattr(torch, "no_grad", contextlib.nullcontext)
        except Exception:
            no_grad_ctx = contextlib.nullcontext

        is_npu = self.device == "NPU"
        max_len = 128 if is_npu else 256
        pad_mode = "max_length" if is_npu else True

        results: List[np.ndarray] = []
        for i in range(0, len(sentences), batch_size):
            batch = list(sentences[i : i + batch_size])
            inputs = self.tokenizer(
                batch,
                padding=pad_mode,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            with no_grad_ctx():
                outputs = self.model(**inputs)
                raw_logits = outputs.logits.squeeze(-1)
                logits = (
                    raw_logits.cpu().numpy()
                    if hasattr(raw_logits, "cpu")
                    else np.asarray(raw_logits)
                )
            if logits.ndim == 0:
                logits = np.array([float(logits)], dtype=np.float32)
            results.append(logits)

        return np.concatenate(results) if results else np.array([], dtype=np.float32)


def get_cross_encoder_device() -> str:
    """Resolve compute device, prioritizing explicit env override, OpenVINO (GPU/NPU), XPU, CUDA, MPS, and CPU."""
    env_device = os.getenv("CROSS_ENCODER_DEVICE")
    if env_device and env_device.strip():
        return env_device.strip()

    backend = os.getenv("CROSS_ENCODER_BACKEND", "").strip().lower()
    if backend == "openvino":
        try:
            import openvino as ov

            core = ov.Core()
            available = set(core.available_devices)
            if "GPU" in available:
                return "GPU"
            if "NPU" in available:
                return "NPU"
            return "CPU"
        except Exception as exc:  # pragma: no cover
            logger.debug("OpenVINO device detection fallback: %s", exc)

    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception as exc:  # pragma: no cover - defensive hardware check
        logger.debug("Device detection fallback to CPU: %s", exc)

    return "cpu"


def _load_cross_encoder(name: str, device: str) -> Any:
    backend = os.getenv("CROSS_ENCODER_BACKEND", "").strip().lower()
    dev_upper = device.upper()

    if backend == "openvino" or dev_upper in {"GPU", "NPU"}:
        try:
            return OpenVINOCrossEncoder(name, device=dev_upper)
        except Exception as exc:
            logger.warning(
                "Failed to initialize OpenVINO CrossEncoder on %s (%s); falling back to PyTorch.",
                device,
                exc,
            )

    from sentence_transformers import CrossEncoder

    return CrossEncoder(name, device=device.lower())


def get_cross_encoder_model(model_name: str | None = None) -> CrossEncoder:
    """Thread-safe lazy-loaded singleton for the Cross-Encoder model."""
    name = (
        model_name or os.getenv("CROSS_ENCODER_MODEL") or DEFAULT_CROSS_ENCODER_MODEL
    ).strip()

    with _model_lock:
        if name in _model_cache:
            return _model_cache[name]

        device = get_cross_encoder_device()
        logger.info(
            "Initializing CrossEncoder model '%s' on device '%s'...", name, device
        )
        model = _load_cross_encoder(name, device)
        _model_cache[name] = model
        return model


def reset_cross_encoder_cache_for_tests() -> None:
    """Helper to clear cached model instances in test fixtures."""
    with _model_lock:
        _model_cache.clear()


@overload
def sigmoid(x: float) -> float: ...


@overload
def sigmoid(x: np.ndarray) -> np.ndarray: ...


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid function."""
    clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _extract_names(raw: Any) -> List[str]:
    """Helper to extract names from list of dicts, strings, or comma-separated values."""
    if not raw:
        return []
    names: List[str] = []
    if isinstance(raw, (list, tuple)):
        for entry in raw:
            if isinstance(entry, dict):
                name = entry.get("name")
            else:
                name = getattr(entry, "name", None)
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
            elif isinstance(entry, str) and entry.strip():
                names.append(entry.strip())
    elif isinstance(raw, dict):
        name = raw.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    elif isinstance(raw, str) and raw.strip():
        names.append(raw.strip())
    return list(dict.fromkeys(names))


def build_candidate_document(item: Dict[str, Any]) -> str:
    """
    Build a dense semantic string for the candidate item covering:
    Title, Year, Media Type, Genres, Directors, Cast, and Overview.
    """
    title = str(item.get("title") or item.get("name") or "").strip()
    year = item.get("release_year")
    media_type = item.get("media_type")
    overview = str(item.get("overview") or "").strip()

    genres = _extract_names(item.get("genres"))
    directors = _extract_names(item.get("directors"))
    cast = _extract_names(item.get("cast"))

    parts: List[str] = []
    header_bits: List[str] = []
    if title:
        header_bits.append(title)
    if year:
        header_bits.append(f"({year})")
    if media_type:
        header_bits.append(f"[{media_type}]")
    if header_bits:
        parts.append(" ".join(header_bits))

    if genres:
        parts.append("Genres: " + ", ".join(genres[:5]))
    if directors:
        parts.append("Directed by " + ", ".join(directors[:3]))
    if cast:
        parts.append("Starring " + ", ".join(cast[:4]))
    if overview:
        parts.append(overview)

    doc = ". ".join(p for p in parts if p).strip()
    if not doc:
        doc = f"Item {item.get('id', 'unknown')}"
    return doc


def format_query_text(
    query: str | None, intent_genres: Sequence[str] | None = None
) -> str:
    """Format query text and optional intent genres for cross-attention."""
    segments: List[str] = []
    if query and query.strip():
        segments.append(query.strip())
    if intent_genres:
        valid_genres = [g.strip() for g in intent_genres if g and g.strip()]
        if valid_genres:
            segments.append("Genres: " + ", ".join(valid_genres[:4]))
    return " | ".join(segments).strip()


def score_query_candidates(
    query: str,
    candidates: Sequence[Dict[str, Any]],
    *,
    model_name: str | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    alpha: float | None = None,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
) -> List[Tuple[int, float, float]]:
    """
    Score candidates against the query using batched joint attention Cross-Encoder.

    Returns:
        List of tuples: (item_id, blended_score, cross_encoder_score)
        sorted in descending order of blended_score.
    """
    if not candidates:
        return []

    cleaned_query = query.strip() if query else ""
    if not cleaned_query:
        # Fall back to base retrieval ranking if no query text exists
        result: List[Tuple[int, float, float]] = []
        for idx, item in enumerate(candidates):
            ident = item.get("id")
            if ident is None:
                continue
            retrieval = float(
                item.get("retrieval_score") or item.get("score") or (1.0 / (1.0 + idx))
            )
            result.append((int(ident), retrieval, retrieval))
        return result

    effective_alpha = (
        _DEFAULT_ALPHA if alpha is None else max(0.0, min(1.0, float(alpha)))
    )
    eval_candidates = list(candidates[:max_candidates])
    pairs: List[Tuple[str, str]] = []
    valid_items: List[Dict[str, Any]] = []

    for item in eval_candidates:
        ident = item.get("id")
        if ident is None:
            continue
        doc_text = build_candidate_document(item)
        pairs.append((cleaned_query, doc_text))
        valid_items.append(item)

    if not pairs:
        return []

    try:
        model = get_cross_encoder_model(model_name)
        raw_scores = model.predict(pairs, batch_size=batch_size, convert_to_numpy=True)
    except Exception as exc:  # pragma: no cover - defensive inference guard
        logger.warning("CrossEncoder scoring failed (%s); using baseline scores.", exc)
        return [
            (
                int(item["id"]),
                float(item.get("retrieval_score") or item.get("score") or 0.0),
                0.0,
            )
            for item in valid_items
        ]

    arr = np.asarray(raw_scores, dtype=np.float32)
    # Calibrate logits with sigmoid to [0, 1]
    calibrated_scores = sigmoid(arr)
    if calibrated_scores.ndim == 0:
        calibrated_scores = np.array([float(calibrated_scores)], dtype=np.float32)

    scored_list: List[Tuple[int, float, float, int]] = []
    for idx, (item, ce_score) in enumerate(zip(valid_items, calibrated_scores)):
        ident = int(item["id"])
        base_rank = int(item.get("original_rank", idx))
        retrieval_val = item.get("retrieval_score")
        if retrieval_val is not None:
            base_score = float(retrieval_val)
        else:
            base_score = 1.0 / (1.0 + float(base_rank))

        # Normalize base_score if greater than 1.0
        if base_score > 1.0:
            norm_base = min(1.0, base_score / 2.0)
        else:
            norm_base = max(0.0, base_score)

        blended = (
            effective_alpha * float(ce_score) + (1.0 - effective_alpha) * norm_base
        )
        scored_list.append(
            (ident, round(blended, 4), round(float(ce_score), 4), base_rank)
        )

    # Sort descending by blended score, breaking ties with original base_rank
    scored_list.sort(key=lambda x: (-x[1], x[3]))
    return [(ident, blended, ce_score) for ident, blended, ce_score, _ in scored_list]
