from __future__ import annotations

import logging
import os
import threading
from typing import Iterable, Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

_model: Any | None = None
_model_cache: dict[str, Any] = {}
_model_lock = threading.Lock()


class OpenVINOEmbeddingModel:
    """OpenVINO-accelerated embedding extractor for Intel Arc GPU, NPU, and CPU."""

    def __init__(self, model_name: str, device: str = "GPU"):
        from pathlib import Path
        from optimum.intel.openvino import OVModelForFeatureExtraction
        from transformers import AutoTokenizer

        hf_model_id = (
            f"sentence-transformers/{model_name}"
            if "/" not in model_name
            else model_name
        )
        self.device = device.upper()
        safe_name = hf_model_id.replace("/", "--")
        cache_dir = Path.home() / ".cache" / "openvino_models" / safe_name

        if (cache_dir / "openvino_model.xml").exists():
            self.tokenizer = AutoTokenizer.from_pretrained(cache_dir)
            if self.device == "NPU":
                self.model = OVModelForFeatureExtraction.from_pretrained(
                    cache_dir, compile=False
                )
                batch_size = int(os.getenv("EMBED_BATCH", "64"))
                self.model.reshape(batch_size, 128)
                self.model.to("NPU")
                self.model.compile()
            else:
                self.model = OVModelForFeatureExtraction.from_pretrained(
                    cache_dir, device=self.device
                )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
            if self.device == "NPU":
                self.model = OVModelForFeatureExtraction.from_pretrained(
                    hf_model_id, export=True, compile=False
                )
                batch_size = int(os.getenv("EMBED_BATCH", "64"))
                self.model.reshape(batch_size, 128)
                self.model.to("NPU")
                self.model.compile()
            else:
                self.model = OVModelForFeatureExtraction.from_pretrained(
                    hf_model_id, export=True, device=self.device
                )
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(cache_dir)
                self.tokenizer.save_pretrained(cache_dir)
            except Exception as exc:
                logger.debug("Could not cache OpenVINO model to %s: %s", cache_dir, exc)

    def encode(
        self,
        texts: Iterable[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, 384), dtype=np.float32)

        import contextlib

        try:
            import torch

            no_grad_ctx = getattr(torch, "no_grad", contextlib.nullcontext)
        except Exception:
            no_grad_ctx = contextlib.nullcontext

        is_npu = self.device == "NPU"
        max_len = 128 if is_npu else 256
        pad_mode = "max_length" if is_npu else True

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=pad_mode,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            with no_grad_ctx():
                outputs = self.model(**inputs)
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]

                if hasattr(attention_mask, "unsqueeze"):
                    mask = (
                        attention_mask.unsqueeze(-1)
                        .expand(token_embeddings.size())
                        .float()
                    )
                    sum_embeddings = (token_embeddings * mask).sum(1)
                    sum_mask = mask.sum(1).clamp(min=1e-9)
                    pooled = sum_embeddings / sum_mask

                    if normalize_embeddings:
                        import torch

                        if hasattr(torch, "nn") and hasattr(torch.nn, "functional"):
                            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

                    arr = (
                        pooled.cpu().numpy()
                        if hasattr(pooled, "cpu")
                        else np.asarray(pooled)
                    )
                else:
                    mask = np.expand_dims(attention_mask, -1).astype(np.float32)
                    sum_embeddings = (token_embeddings * mask).sum(axis=1)
                    sum_mask = np.clip(mask.sum(axis=1), 1e-9, None)
                    pooled = sum_embeddings / sum_mask
                    if normalize_embeddings:
                        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                        norms[norms == 0] = 1e-12
                        pooled = pooled / norms
                    arr = np.asarray(pooled, dtype=np.float32)

                all_embeddings.append(arr.astype(np.float32))

        return (
            np.vstack(all_embeddings)
            if all_embeddings
            else np.zeros((0, 384), dtype=np.float32)
        )


def get_embedding_device() -> str:
    """Resolve compute device, prioritizing explicit env override, OpenVINO (GPU/NPU), CUDA, XPU, and CPU."""
    env_device = os.getenv("EMBEDDING_DEVICE") or os.getenv("DEVICE")
    if env_device and env_device.strip():
        return env_device.strip()

    backend = os.getenv("EMBEDDING_BACKEND", "").strip().lower()
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

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except Exception as exc:  # pragma: no cover
        logger.debug("Device detection fallback to CPU: %s", exc)

    return "cpu"


def get_model(device: str | None = None) -> Any:
    global _model
    target_device = device or get_embedding_device()
    dev_key = target_device.upper()
    backend = os.getenv("EMBEDDING_BACKEND", "").strip().lower()

    with _model_lock:
        if dev_key in _model_cache:
            return _model_cache[dev_key]

        if backend == "openvino" or dev_key in {"GPU", "NPU"}:
            try:
                model = OpenVINOEmbeddingModel(DEFAULT_MODEL, device=dev_key)
                _model_cache[dev_key] = model
                if _model is None:
                    _model = model
                return model
            except Exception as exc:
                logger.warning(
                    "Failed to initialize OpenVINO Embedding model on %s (%s); falling back to PyTorch.",
                    target_device,
                    exc,
                )

        if _model is None:
            from sentence_transformers import (
                SentenceTransformer as _SentenceTransformer,
            )

            dev = target_device.lower()
            if dev in {"gpu", "npu"}:
                dev = "cpu"
            _model = _SentenceTransformer(DEFAULT_MODEL, device=dev)

        _model_cache[dev_key] = _model
        return _model


def reset_embedding_cache_for_tests() -> None:
    global _model
    with _model_lock:
        _model = None
        _model_cache.clear()


def encode_texts(texts: Iterable[str], device: str | None = None) -> np.ndarray:
    """
    Returns float32 numpy array shape (N, 384) for MiniLM, L2-normalized row-wise.
    """
    texts = list(texts)
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    model = get_model(device=device) if device is not None else get_model()
    emb = model.encode(
        texts,
        batch_size=int(os.getenv("EMBED_BATCH", "64")),
        normalize_embeddings=True,  # L2-normalize for cosine
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return emb.astype("float32")
