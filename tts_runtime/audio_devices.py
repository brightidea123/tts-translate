"""Fresh input discovery and capture preflight, without model imports."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
import math
from pathlib import Path
import subprocess


def input_devices(sounddevice) -> list[dict]:
    devices = list(sounddevice.query_devices())
    default = sounddevice.default.device[0]
    names = Counter(str(info["name"]) for info in devices if info["max_input_channels"] > 0)
    return [
        {"id": str(info["name"]), "index": index, "name": str(info["name"]),
         "max_input_channels": int(info["max_input_channels"]),
         "default_samplerate": float(info["default_samplerate"]),
         "default": index == default, "ambiguous": names[str(info["name"])] > 1}
        for index, info in enumerate(devices) if info["max_input_channels"] > 0
    ]


def fresh_input_devices(python_executable: str) -> list[dict]:
    # PortAudio caches the device graph at initialization. Refresh in a new
    # process; terminating/reinitializing it here could interrupt live playback.
    result = subprocess.run(
        [python_executable, "-m", "tts_runtime.audio_devices"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        timeout=8, check=True,
    )
    payload = json.loads(result.stdout)
    if payload.get("error"):
        raise RuntimeError(payload["error"])
    if not isinstance(payload.get("devices"), list):
        raise RuntimeError("Invalid audio input inventory")
    return payload["devices"]


def checked_input_device(sounddevice, requested, *, samplerate: int, channels: int, dtype: str) -> dict:
    """Resolve the selection in this capture process; never substitute a source."""
    try:
        info = sounddevice.query_devices(requested, kind="input")
        if info["max_input_channels"] < channels:
            raise ValueError(f"{info['name']!r} has no usable {channels}-channel input")
        sounddevice.check_input_settings(device=info["index"], samplerate=samplerate,
                                         channels=channels, dtype=dtype)
        return dict(info)
    except Exception as exc:
        try:
            available = "; ".join(f"{item['index']}: {item['name']}" for item in input_devices(sounddevice))
        except Exception:
            available = "device inventory unavailable"
        selection = "system default" if requested is None else repr(requested)
        raise RuntimeError(
            f"Cannot capture from {selection}: {exc}. "
            f"Available inputs: {available or 'none'}. "
            "Refresh Audio input in the web UI and select the intended input by name, "
            "or run --list-devices and use --input-device 'exact input name'. "
            "Device numbers can change after audio devices reconnect."
        ) from exc


def describe_input_device(info: Mapping, requested, *, samplerate: int, channels: int, dtype: str,
                          default_index=None, hostapi_info: Mapping | None = None) -> dict:
    """Describe an already-resolved input without querying or changing hardware.

    Call only after capture preflight, using its returned device dictionary.
    Optional default/host-API metadata must come from that capture process;
    absent metadata remains unknown instead of implying a physical microphone.
    Stream format and device capabilities are reported separately.
    """
    info = info if isinstance(info, Mapping) else {}
    hostapi_info = hostapi_info if isinstance(hostapi_info, Mapping) else {}

    def number(value, *, minimum=0):
        if isinstance(value, bool):
            return None
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) and value >= minimum else None

    def integer(value):
        value = number(value)
        return int(value) if value is not None and value.is_integer() else None

    def name(value):
        return value if isinstance(value, str) and value.strip() else None

    index = integer(info.get("index"))
    default = integer(default_index)
    stream_rate = number(samplerate, minimum=1)
    stream_channels = integer(channels)
    return {
        "backend": "sounddevice",
        "requested": requested if not isinstance(requested, bool) and isinstance(requested, (str, int)) else None,
        "requested_default": requested is None,
        "name": name(info.get("name")),
        "index": index,
        "default_input_index": default,
        "is_default": index == default if index is not None and default is not None else None,
        "hostapi_index": integer(info.get("hostapi")),
        "hostapi_name": name(hostapi_info.get("name")),
        "max_input_channels": integer(info.get("max_input_channels")),
        "default_samplerate": number(info.get("default_samplerate"), minimum=1),
        "sample_rate": stream_rate,
        "channels": stream_channels if stream_channels is not None and stream_channels >= 1 else None,
        "dtype": name(dtype),
    }


def main() -> int:
    try:
        import sounddevice
        payload = {"devices": input_devices(sounddevice), "error": ""}
    except Exception as exc:
        payload = {"devices": [], "error": str(exc)}
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
