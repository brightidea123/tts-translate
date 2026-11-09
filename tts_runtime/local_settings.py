"""Read this checkout's named input preference without touching audio hardware."""
from __future__ import annotations

import json
from pathlib import Path
import re
import unicodedata


def load_input_preference(path: str | Path) -> str | None:
    """Return a saved device name, or None for an absent/default preference.

    Invalid files fail explicitly: silently reverting to the system default can
    capture a different source. Explicit CLI and web selections take precedence
    over this preference in the caller.
    """
    path = Path(path)
    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read local settings {path}: {exc}") from exc
    try:
        settings = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in local settings {path}: {exc.msg}") from exc
    if not isinstance(settings, dict) or set(settings) - {"input_device"}:
        raise ValueError(f"Local settings {path} must be an object containing only input_device")
    device = settings.get("input_device")
    if device is None:
        return None
    if (not isinstance(device, str) or not device.strip() or len(device) > 256
            or any(unicodedata.category(char) == "Cc" for char in device)):
        raise ValueError(f"Local settings {path}: input_device must be null or a nonempty device name of at most 256 characters without control characters")
    device = device.strip()
    if re.fullmatch(r"[+-]?\d+", device):
        raise ValueError(f"Local settings {path}: save the input device name, not a numeric index that can change after reconnecting")
    return device
