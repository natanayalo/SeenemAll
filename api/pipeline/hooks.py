from __future__ import annotations

import sys
from typing import Any


def get_hook(name: str, default: Any) -> Any:
    """
    Look up a symbol from api.routes.recommend dynamically.
    If it has been monkeypatched or defined in tests, returns that value.
    Otherwise returns default.
    """
    mod = sys.modules.get("api.routes.recommend")
    if mod is not None and hasattr(mod, name):
        return getattr(mod, name)
    return default
