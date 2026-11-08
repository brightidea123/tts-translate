"""Fresh input discovery and capture preflight, without model imports."""
from __future__ import annotations

from collections import Counter
import json
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
