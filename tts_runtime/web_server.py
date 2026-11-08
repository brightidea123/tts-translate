"""Loopback-only HTTP/WebSocket bridge for the local transcription controller.

No model or capture code runs in the ASGI event loop. The injectable controller
admits commands quickly; potentially blocking device/state calls use the worker
thread pool. This module never opens a microphone or starts a model itself.
"""
from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator
from starlette.concurrency import run_in_threadpool

from translation_prompts import normalize_translation_language


ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = {"AR", "HE", "DE", "TR", "FR", "RU", "EN", "ES"}
KNOWN_ASSETS = {"/": "index.html", "/index.html": "index.html",
                "/app.js": "app.js", "/ui-interactions.js": "ui-interactions.js", "/styles.css": "styles.css"}


def _loopback(value: str | None) -> bool:
    if not value:
        return False
    hostname = value.strip("[]").casefold()
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _origin_error(connection) -> str | None:
    try:
        host = urlsplit("//" + connection.headers.get("host", ""))
        host_port = host.port
    except ValueError:
        return "Invalid Host header"
    if not _loopback(host.hostname) or host.username or host.password or host.path or host.query or host.fragment:
        return "This interface accepts only localhost requests"
    if connection.client is None or not _loopback(connection.client.host):
        return "This interface accepts only loopback clients"
    origin_header = connection.headers.get("origin")
    if origin_header:
        try:
            origin = urlsplit(origin_header)
            origin_port = origin.port or (443 if origin.scheme == "https" else 80)
        except ValueError:
            return "Invalid Origin header"
        expected_scheme = "https" if connection.scope.get("scheme") in {"https", "wss"} else "http"
        expected_port = host_port or (443 if expected_scheme == "https" else 80)
        if (origin.scheme != expected_scheme or origin.hostname != host.hostname
                or origin_port != expected_port or origin.username or origin.password
                or origin.path not in {"", "/"} or origin.query or origin.fragment):
            return "Cross-origin requests are not allowed"
    return None


class StartConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_language: StrictStr = "TR"
    target_language: StrictStr = "EN"
    input_device: StrictStr | StrictInt | None = None
    debug: StrictBool = False

    @field_validator("source_language", "target_language")
    @classmethod
    def language(cls, value: str) -> str:
        if len(value) > 32:
            raise ValueError("Language code is too long")
        normalized = normalize_translation_language(value).upper()
        if normalized not in LANGUAGES:
            raise ValueError("Choose AR, HE, DE, TR, FR, RU, EN, or ES")
        return normalized

    @field_validator("input_device")
    @classmethod
    def device(cls, value):
        if isinstance(value, str):
            if len(value) > 256 or any(ord(char) < 32 for char in value):
                raise ValueError("Input device name must be at most 256 characters without control characters")
            return value.strip() or None
        if isinstance(value, int) and not -1 <= value <= 2147483647:
            raise ValueError("Input device index is out of range")
        return value

    @model_validator(mode="after")
    def different_languages(self):
        if self.source_language == self.target_language:
            raise ValueError("Source and target languages must differ")
        return self


class Reply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: StrictStr = Field(min_length=1, max_length=10000)

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Reply text must not be blank")
        return value


class Mark(BaseModel):
    model_config = ConfigDict(extra="forbid")
    utterance_id: StrictInt | None = Field(default=None, ge=1)
    reason: StrictStr = Field(default="manual-review", min_length=1, max_length=1000)


class SelectionTranslation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: StrictStr = Field(min_length=1, max_length=1500)
    source_language: StrictStr
    target_language: StrictStr
    selection_id: StrictStr = Field(min_length=1, max_length=128)

    @field_validator("source_language", "target_language")
    @classmethod
    def language(cls, value: str) -> str:
        return StartConfig.language(value)

    @model_validator(mode="after")
    def valid_selection(self):
        from .selection_requests import SelectionRequests
        SelectionRequests.validate(self.model_dump())
        return self


async def _call(controller, method: str, *args):
    result = await run_in_threadpool(getattr(controller, method), *args)
    return await result if inspect.isawaitable(result) else result


class _Snapshots:
    """Serialize snapshot reads and reuse each poll across browser clients."""
    def __init__(self, controller):
        self.controller = controller
        self.lock = asyncio.Lock()
        self.fingerprint = None
        self.event = None
        self.last_query = 0.0
        self.version = 0

    async def current(self, *, force: bool = False) -> dict:
        async with self.lock:
            now = time.monotonic()
            if self.event is not None and not force and now - self.last_query < 0.08:
                return self.event
            state = await _call(self.controller, "state")
            if not isinstance(state, dict):
                raise RuntimeError("Controller state must be a JSON object")
            fingerprint = json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False)
            self.last_query = time.monotonic()
            if fingerprint != self.fingerprint:
                self.version += 1
                self.fingerprint = fingerprint
                # Freeze mutable controller data before yielding to another
                # request; an emitted version always describes one snapshot.
                frozen = json.loads(fingerprint)
                self.event = {"type": "state", "session_id": frozen.get("session_id"),
                              "version": self.version, "state": frozen}
            return self.event


def create_app(controller, assets_dir: str | Path = ROOT / "web") -> FastAPI:
    app = FastAPI(title="Local TTS Translate", docs_url=None, redoc_url=None, openapi_url=None)
    assets = Path(assets_dir).resolve()
    snapshots = _Snapshots(controller)
    app.state.controller = controller
    app.state.snapshots = snapshots

    @app.middleware("http")
    async def localhost(request: Request, call_next):
        error = _origin_error(request)
        if error:
            return JSONResponse(status_code=403, content={"detail": error})
        length = request.headers.get("content-length")
        if length:
            try:
                if int(length) > 262144:
                    return JSONResponse(status_code=413, content={"detail": "Request body exceeds 256 KiB"})
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, error):
        details = []
        for item in error.errors()[:5]:
            location = ".".join(str(value) for value in item.get("loc", ()) if value != "body")
            details.append(f"{location}: {item['msg']}" if location else item["msg"])
        return JSONResponse(status_code=400, content={"detail": "; ".join(details)})

    async def command(method: str, *args):
        try:
            result = await _call(controller, method, *args)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Runtime {method} failed: {exc}") from exc
        if result is None:
            result = {"ok": True}
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Controller command returned an invalid response")
        if result.get("ok") is False or result.get("accepted") is False:
            return JSONResponse(status_code=409, content=result)
        return result

    @app.get("/api/state")
    async def state():
        try:
            return (await snapshots.current(force=True))["state"]
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Cannot read runtime state: {exc}") from exc

    @app.get("/api/devices")
    async def devices():
        try:
            found = await _call(controller, "devices")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Cannot list audio devices: {exc}") from exc
        if isinstance(found, dict) and isinstance(found.get("devices"), list):
            return found
        if not isinstance(found, list):
            raise HTTPException(status_code=500, detail="Controller devices must be a list")
        error = getattr(controller, "device_error", "")
        return {"devices": found, **({"error": error} if error else {})}

    @app.post("/api/start")
    async def start(config: StartConfig):
        return await command("start", config.model_dump())

    @app.post("/api/stop")
    async def stop():
        return await command("stop")

    @app.post("/api/reply")
    async def reply(body: Reply):
        return await command("reply", body.text)

    @app.post("/api/flush")
    async def flush():
        return await command("flush")

    @app.post("/api/mark")
    async def mark(body: Mark | None = None):
        return await command("mark") if body is None else await command("mark", body.utterance_id, body.reason)

    @app.post("/api/selection-translation")
    async def selection_translation(body: SelectionTranslation):
        return await command("selection_translation", body.model_dump())

    @app.get("/api/selection-translation/{lookup_id}")
    async def selection_translation_result(lookup_id: str):
        if len(lookup_id) != 32 or any(char not in "0123456789abcdef" for char in lookup_id):
            raise HTTPException(status_code=404, detail="Selection translation is no longer available")
        result = await _call(controller, "selection_translation_result", lookup_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Selection translation is no longer available")
        return result

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket):
        if _origin_error(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        previous_version = None
        try:
            while True:
                event = await snapshots.current()
                if event["version"] != previous_version:
                    await websocket.send_json(event)
                    previous_version = event["version"]
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=0.1)
                    if message["type"] == "websocket.disconnect":
                        break
                    if message.get("text") == "ping":
                        await websocket.send_json({"type": "pong"})
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            pass
        except Exception:
            await websocket.close(code=1011, reason="Runtime state is unavailable")

    async def asset(request: Request):
        filename = KNOWN_ASSETS.get(request.url.path)
        if filename is None:
            raise HTTPException(status_code=404, detail="Not found")
        path = (assets / filename).resolve()
        if path.parent != assets or not path.is_file():
            raise HTTPException(status_code=503, detail="Local web assets are not available")
        return FileResponse(path)

    for url in KNOWN_ASSETS:
        app.add_api_route(url, asset, methods=["GET"], include_in_schema=False)
    return app


class WebInterface:
    """Start/stop the loopback Uvicorn server without blocking the UI loop."""
    def __init__(self, controller=None, host: str = "127.0.0.1", port: int = 8765,
                 *, assets_dir: str | Path = ROOT / "web"):
        self.controller, self.host, self.port = controller, host, port
        self.assets_dir = Path(assets_dir)
        self.server = self.thread = None
        self.error = ""
        self.url: str | None = None

    def start(self, controller=None, host: str | None = None, port: int | None = None,
              *, timeout: float = 1.0) -> str:
        if self.thread is not None and self.thread.is_alive():
            if self.url:
                return self.url
            raise RuntimeError("Web server is already starting")
        self.controller = self.controller if controller is None else controller
        self.host = self.host if host is None else host
        self.port = self.port if port is None else port
        if self.controller is None:
            raise ValueError("A runtime controller is required")
        if not _loopback(self.host):
            raise ValueError("Web interface must bind to a loopback address")
        if not isinstance(self.port, int) or isinstance(self.port, bool) or not 0 <= self.port <= 65535:
            raise ValueError("Web port must be an integer from 0 to 65535")
        if timeout <= 0:
            raise ValueError("Startup timeout must be positive")
        import uvicorn
        app = create_app(self.controller, self.assets_dir)
        configuration = uvicorn.Config(app, host=self.host.strip("[]"), port=self.port,
                                       log_config=None, log_level="warning", access_log=False,
                                       lifespan="off")
        self.server = uvicorn.Server(configuration)
        self.error, self.url = "", None

        def serve():
            try:
                self.server.run()
            except BaseException as exc:
                self.error = f"{type(exc).__name__}: {exc}"

        self.thread = threading.Thread(target=serve, name="local-web-interface", daemon=True)
        self.thread.start()
        deadline = time.monotonic() + timeout
        waiter = threading.Event()
        while time.monotonic() < deadline:
            if self.server.started and self.thread.is_alive():
                bound_port = self.port
                servers = getattr(self.server, "servers", ())
                if servers and servers[0].sockets:
                    bound_port = int(servers[0].sockets[0].getsockname()[1])
                hostname = f"[{self.host.strip('[]')}]" if ":" in self.host else self.host
                self.url = f"http://{hostname}:{bound_port}"
                print(f"Web UI: {self.url}", flush=True)
                return self.url
            if not self.thread.is_alive():
                raise RuntimeError(f"Cannot start web interface: {self.error or 'server exited before listening'}")
            waiter.wait(min(0.01, max(0.0, deadline - time.monotonic())))
        self.close(timeout=0.05)
        raise RuntimeError(f"Web interface did not start within {timeout:g}s")

    def close(self, timeout: float = 2.0) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=max(0.0, timeout))
            if self.thread.is_alive() and self.server is not None:
                self.server.force_exit = True
        self.url = None
