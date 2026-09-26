from __future__ import annotations

from pathlib import Path
from typing import Any


def save_tokenizer_safely(tokenizer: Any, cache_dir: Path) -> None:
    """Save a tokenizer after rejecting unsafe named chat templates.

    Transformers versions before 5.10.0 allow chat template names to escape
    the target directory during ``save_pretrained`` (CVE-2026-9856). OpenVINO
    currently requires Transformers below 5.6, so apply the upstream path
    validation at the application boundary until that constraint is lifted.
    """

    chat_template = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_template, dict):
        template_dir = (cache_dir / "additional_chat_templates").resolve()
        for template_name in chat_template:
            template_path = (template_dir / f"{template_name}.jinja").resolve()
            if template_path.parent != template_dir:
                raise ValueError(f"Invalid chat template name: {template_name!r}")

    tokenizer.save_pretrained(cache_dir)
