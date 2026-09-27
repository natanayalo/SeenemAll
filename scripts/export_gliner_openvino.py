"""Export the pinned GLiNER checkpoint as the OpenVINO IR used by the API."""

from __future__ import annotations

import json
import os
from pathlib import Path

from gliner import GLiNER

DEFAULT_MODEL = "urchade/gliner_small-v2.1"
DEFAULT_REVISION = "4e091416cf7c3481db542c2a3d26156916f3a47f"
DEFAULT_OUTPUT_DIR = "models/gliner_small_ov"


def main() -> None:
    model_id = os.getenv("FAST_INTENT_MODEL", DEFAULT_MODEL)
    revision = os.getenv("FAST_INTENT_MODEL_REVISION", DEFAULT_REVISION)
    output_dir = Path(
        os.getenv("FAST_INTENT_OPENVINO_DIR", DEFAULT_OUTPUT_DIR)
    ).resolve()

    model = GLiNER.from_pretrained(model_id, revision=revision)
    artifacts = model.export_to_openvino(output_dir, compress_to_fp16=True)

    metadata = {
        "model_id": model_id,
        "revision": revision,
        "runtime": "openvino",
        "precision": "fp16-compressed",
    }
    (output_dir / "model_source.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported {model_id}@{revision} to {artifacts['openvino_path']}")


if __name__ == "__main__":
    main()
