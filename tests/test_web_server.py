from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tts_runtime.web_server import WebInterface, create_app


BASE_URL = "http://127.0.0.1:8765"


class Controller:
    def __init__(self):
        self.lock = threading.Lock()
        self.current = {"session_id": "session-one", "running": False, "status": "idle", "transcript": []}
        self.calls = []

    def state(self):
        with self.lock:
            return deepcopy(self.current)

    def change(self, **kwargs):
        with self.lock:
            self.current.update(kwargs)

    def devices(self):
        return [{"index": 1, "name": "LitLink Audio Bridge", "input_channels": 2}]

    def start(self, config):
        self.calls.append(("start", config))
        return {"accepted": True, "message": "Start queued"}

    def stop(self):
        self.calls.append(("stop",))
        return {"accepted": True}

    def reply(self, text):
        self.calls.append(("reply", text))
        return {"accepted": True}

    def flush(self):
        self.calls.append(("flush",))
        return {"accepted": True}

    def mark(self, utterance_id=None, reason="manual-review"):
        self.calls.append(("mark", utterance_id, reason))
        return {"accepted": True}

    def selection_translation(self, config):
        self.calls.append(("selection", config))
        return {"accepted": True, "lookup_id": "a" * 32, "status": "pending", **config}

    def selection_translation_result(self, identity):
        return {"status": "complete", "translation": "مرحبا", "lookup_id": identity} if identity == "a" * 32 else None


class WebServerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.assets = Path(self.directory.name)
        for name, text in (("index.html", "<html>Local translator</html>"), ("app.js", "/* local code */"), ("ui-interactions.js", "/* interactions */"), ("styles.css", "body{color:black}")):
            (self.assets / name).write_text(text, encoding="utf-8")
        self.controller = Controller()
        self.app = create_app(self.controller, self.assets)

    def client(self, **kwargs):
        return TestClient(self.app, base_url=BASE_URL, client=("127.0.0.1", 50000), **kwargs)

    def test_device_discovery_error_is_returned_to_the_page(self):
        self.controller.device_error = "No audio inputs found; check LitLink's input route"
        with self.client() as client, patch.object(self.controller, "devices", return_value=[]):
            payload = client.get("/api/devices").json()
            self.assertEqual(payload["devices"], [])
            self.assertEqual(payload["error"], self.controller.device_error)

    def test_static_state_devices_and_all_command_boundaries(self):
        with self.client() as client:
            self.assertIn("Local translator", client.get("/").text)
            self.assertEqual(client.get("/app.js").status_code, 200)
            self.assertEqual(client.get("/ui-interactions.js").status_code, 200)
            self.assertEqual(client.get("/styles.css").status_code, 200)
            self.assertEqual(client.get("/tts-transcribe.py").status_code, 404)
            self.assertEqual(client.get("/api/state").json(), self.controller.current)
            self.assertEqual(client.get("/api/devices").json()["devices"][0]["name"], "LitLink Audio Bridge")
            response = client.post("/api/start", json={"source_language": "ar", "target_language": "Hebrew", "input_device": "LitLink Audio Bridge", "debug": True}, headers={"Origin": BASE_URL})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(self.controller.calls[-1], ("start", {"source_language": "AR", "target_language": "HE", "input_device": "LitLink Audio Bridge", "debug": True}))
            phrase = "שלום مرحبا Привет\nune réponse"
            self.assertEqual(client.post("/api/reply", json={"text": phrase}).status_code, 200)
            self.assertEqual(self.controller.calls[-1], ("reply", phrase))
            for method in ("stop", "flush", "mark"):
                self.assertEqual(client.post(f"/api/{method}").status_code, 200)
            self.assertEqual(client.post("/api/mark", json={"utterance_id": 7, "reason": "wrong words שלום"}).status_code, 200)
            self.assertEqual(self.controller.calls[-1], ("mark", 7, "wrong words שלום"))
            self.assertEqual(client.get("/api/state").headers["cache-control"], "no-store")

    def test_invalid_payloads_are_actionable_400_without_admitting_commands(self):
        cases = [
            ("start", {"source_language": "unknown", "target_language": "HE"}),
            ("start", {"source_language": "HE", "target_language": "HE"}),
            ("start", {"debug": "yes"}),
            ("start", {"input_device": True}),
            ("start", {"input_device": "bad\x00device"}),
            ("start", {"unexpected": "field"}),
            ("reply", {"text": "  \n"}),
            ("reply", {"text": "x" * 10001}),
            ("reply", {"text": 123}),
            ("mark", {"utterance_id": -1}),
        ]
        with self.client() as client:
            for endpoint, payload in cases:
                with self.subTest(endpoint=endpoint, payload=str(payload)[:60]):
                    response = client.post(f"/api/{endpoint}", json=payload)
                    self.assertEqual(response.status_code, 400)
                    self.assertTrue(response.json()["detail"])
            self.assertEqual(client.post("/api/start", content="{", headers={"Content-Type": "application/json"}).status_code, 400)
        self.assertEqual(self.controller.calls, [])

    def test_selection_translation_contract_directions_limits_and_missing_results(self):
        with self.client() as client:
            body = {"text": "שלום", "source_language": "he", "target_language": "Arabic", "selection_id": "chosen"}
            reply = client.post("/api/selection-translation", json=body)
            self.assertEqual(reply.status_code, 200, reply.text)
            self.assertEqual(self.controller.calls[-1][1]["source_language"], "HE")
            self.assertEqual(self.controller.calls[-1][1]["target_language"], "AR")
            self.assertEqual(client.get("/api/selection-translation/" + "a" * 32).json()["translation"], "مرحبا")
            self.assertEqual(client.get("/api/selection-translation/" + "b" * 32).status_code, 404)
            self.assertEqual(client.get("/api/selection-translation/bad").status_code, 404)
            for override in [{"text": "x" * 1501}, {"target_language": "HE"}, {"selection_id": "\n"}]:
                self.assertEqual(client.post("/api/selection-translation", json={**body, **override}).status_code, 400)
            self.assertEqual(client.post("/api/selection-translation", json=body, headers={"Origin": "https://foreign.example"}).status_code, 403)

    def test_foreign_origin_rebinding_host_and_remote_peer_are_rejected(self):
        with self.client() as client:
            for headers in ({"Origin": "https://example.com"}, {"Origin": "http://127.0.0.1:9999"}, {"Origin": "http://[broken"}, {"Host": "attacker.example"}, {"Host": "[broken"}):
                response = client.post("/api/stop", headers=headers)
                self.assertEqual(response.status_code, 403, response.text)
        with TestClient(self.app, base_url=BASE_URL, client=("192.0.2.10", 54321)) as client:
            self.assertEqual(client.post("/api/stop").status_code, 403)
        self.assertEqual(self.controller.calls, [])

    def test_controller_rejection_and_exception_remain_visible(self):
        with self.client() as client:
            with patch.object(self.controller, "reply", return_value={"accepted": False, "message": "queue full"}):
                response = client.post("/api/reply", json={"text": "hello"})
                self.assertEqual(response.status_code, 409)
                self.assertEqual(response.json()["message"], "queue full")
            with patch.object(self.controller, "start", side_effect=RuntimeError("already running")):
                response = client.post("/api/start", json={})
                self.assertEqual(response.status_code, 400)
                self.assertIn("already running", response.json()["detail"])

    def test_websocket_initial_snapshot_changes_and_session_versioning(self):
        with self.client() as client:
            with client.websocket_connect("ws://127.0.0.1:8765/ws", headers={"Origin": BASE_URL}) as websocket:
                first = websocket.receive_json()
                self.assertEqual(first["type"], "state")
                self.assertEqual(first["session_id"], "session-one")
                self.assertEqual(first["state"]["status"], "idle")
                websocket.send_text("ping")
                self.assertEqual(websocket.receive_json(), {"type": "pong"})
                self.controller.change(status="listening", running=True)
                second = websocket.receive_json()
                self.assertEqual(second["state"]["status"], "listening")
                self.assertGreater(second["version"], first["version"])
                self.controller.change(session_id="session-two", transcript=[{"source": "שלום"}])
                third = websocket.receive_json()
                self.assertEqual(third["session_id"], "session-two")
                self.assertEqual(third["state"]["transcript"][0]["source"], "שלום")
                self.assertGreater(third["version"], second["version"])
                self.assertEqual(first["state"]["status"], "idle")

    def test_websocket_rejects_foreign_browser_origin(self):
        with self.client() as client:
            with self.assertRaises(WebSocketDisconnect) as caught:
                with client.websocket_connect("ws://127.0.0.1:8765/ws", headers={"Origin": "https://example.com"}):
                    pass
            self.assertEqual(caught.exception.code, 1008)

    def test_slow_device_query_does_not_block_the_event_loop(self):
        entered, release = threading.Event(), threading.Event()

        def devices():
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test query was not released")
            return []

        with self.client() as client, patch.object(self.controller, "devices", side_effect=devices), ThreadPoolExecutor(max_workers=2) as pool:
            query = pool.submit(client.get, "/api/devices")
            try:
                self.assertTrue(entered.wait(1))
                state = pool.submit(client.get, "/api/state")
                self.assertEqual(state.result(timeout=1).json()["status"], "idle")
            finally:
                release.set()
            self.assertEqual(query.result(timeout=1).status_code, 200)


class WebInterfaceTests(unittest.TestCase):
    def test_only_loopback_binding_is_allowed(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            WebInterface(Controller(), host="0.0.0.0").start()

    def test_url_is_announced_only_after_server_is_serving_and_close_is_bounded(self):
        class Server:
            def __init__(self, config):
                self.started = self.should_exit = self.force_exit = False
                self.servers = [SimpleNamespace(sockets=[SimpleNamespace(getsockname=lambda: ("127.0.0.1", 9876))])]

            def run(self):
                self.started = True
                while not self.should_exit and not self.force_exit:
                    threading.Event().wait(0.005)

        with patch("uvicorn.Server", Server), redirect_stdout(io.StringIO()) as output:
            interface = WebInterface(Controller(), port=0)
            try:
                self.assertEqual(interface.start(), "http://127.0.0.1:9876")
                self.assertIn("Web UI: http://127.0.0.1:9876", output.getvalue())
                self.assertTrue(interface.thread.is_alive())
            finally:
                interface.close(timeout=0.5)
            self.assertFalse(interface.thread.is_alive())

    def test_startup_failure_never_announces_a_nonserving_url(self):
        class Server:
            def __init__(self, _config):
                self.started = False
                self.should_exit = False

            def run(self):
                raise OSError("port already in use")

        with patch("uvicorn.Server", Server), redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError, "port already in use"):
                WebInterface(Controller()).start(timeout=0.1)
            self.assertNotIn("Web UI:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
