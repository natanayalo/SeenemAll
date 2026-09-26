from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from api.core.model_security import save_tokenizer_safely


def test_save_tokenizer_allows_plain_chat_template_names(tmp_path):
    tokenizer = MagicMock()
    tokenizer.chat_template = {
        "default": "{{ messages }}",
        "tool_use": "{{ tools }}",
    }

    save_tokenizer_safely(tokenizer, tmp_path)

    tokenizer.save_pretrained.assert_called_once_with(tmp_path)


@pytest.mark.parametrize("template_name", ["../../PWNED", "../outside", "/tmp/PWNED"])
def test_save_tokenizer_rejects_path_traversal(tmp_path, template_name):
    tokenizer = MagicMock()
    tokenizer.chat_template = {template_name: "attacker content"}

    with pytest.raises(ValueError, match="Invalid chat template name"):
        save_tokenizer_safely(tokenizer, tmp_path)

    tokenizer.save_pretrained.assert_not_called()


def test_save_tokenizer_allows_tokenizers_without_chat_templates(tmp_path):
    tokenizer = MagicMock(spec=["save_pretrained"])

    save_tokenizer_safely(tokenizer, tmp_path)

    tokenizer.save_pretrained.assert_called_once_with(tmp_path)
