"""Pinned Ollama decision models through the native System One endpoint."""

import hashlib
import json
import os
import urllib.request
from typing import Any

from evaluation.judge.systemone import SystemOneJudgeAdapter


MODEL_TAGS = {
    "bespoke-nimble-9b": "nimble:latest",
}


class OllamaJudgeAdapter(SystemOneJudgeAdapter):
    """Never pull weights; reject replaced artifacts or changed server versions."""

    def __init__(
        self,
        name: str,
        tag: str,
        metadata: dict,
        version: str,
        url: str,
        evidence_version: str | None = None,
    ):
        digest = metadata.get("digest", "unavailable")
        super().__init__(
            name,
            checkpoint_revision=digest,
            tokenizer_revision=digest,
            quantization=metadata.get("details", {}).get(
                "quantization_level", "unavailable"
            ),
            runtime="systemone_ollama",
            evidence_version=evidence_version,
        )
        self.service_url = url.rstrip("/")
        self.endpoint_url = self.service_url + "/v1/systemone"
        self.service_model = tag
        self.runtime_version = version
        self.timeout_seconds = float(os.environ.get("OLLAMA_JUDGE_TIMEOUT", "120"))
        self.api_key = os.environ.get("OLLAMA_JUDGE_API_KEY")

    def _get(self, path: str) -> Any:
        request = urllib.request.Request(
            self.service_url + path, headers=self._headers()
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.load(response)

    def is_available(self) -> bool:
        try:
            if self._get("/api/version")["version"] != self.runtime_version:
                return False
            return any(
                m.get("name") == self.service_model
                and m.get("digest") == self.checkpoint_revision
                and self.checkpoint_revision != "unavailable"
                and "decision" in m.get("capabilities", [])
                for m in self._get("/api/tags")["models"]
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False

    def qualification_fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(
                [super().qualification_fingerprint(), self.runtime_version],
            ).encode()
        ).hexdigest()


def discover_ollama_judges(
    evidence_version: str | None = None,
) -> dict[str, OllamaJudgeAdapter]:
    """Freeze discovery once for a run so tags cannot silently change provenance."""
    url = os.environ.get("OLLAMA_JUDGE_URL", "http://127.0.0.1:11434").rstrip("/")
    probe = OllamaJudgeAdapter("discovery", "discovery", {}, "unknown", url)
    try:
        version = probe._get("/api/version")["version"]
        models = {m["name"]: m for m in probe._get("/api/tags")["models"]}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        version, models = "unavailable", {}
    return {
        name: OllamaJudgeAdapter(
            name, tag, models.get(tag, {}), version, url, evidence_version
        )
        for name, tag in MODEL_TAGS.items()
    }
