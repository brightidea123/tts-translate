"""Real child JSONL + local ASGI boundary; fake translation, no devices/GPU."""
from pathlib import Path
import sys
import tempfile
import time
import unittest

from fastapi.testclient import TestClient
from tts_runtime.web_controller import WebController
from tts_runtime.web_server import create_app


ROOT = Path(__file__).resolve().parents[1]


class SelectionEndToEndTests(unittest.TestCase):
    def test_http_selection_reaches_real_headless_pipe_with_cache_and_no_reply_side_effect(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            child = directory / "core.py"
            child.write_text("import sys\nsys.path.insert(0, " + repr(str(ROOT)) + ")\n" + '''
from tts_runtime.web_transport import HeadlessTransport
state = {"session_id":"fixture-session", "transcript":[], "replies":[], "lookup_count":0}
def command(packet):
    if packet["action"] == "lookup":
        state["lookup_count"] += 1
        bridge.send_packet({"type":"lookup_result", "lookup_id":packet["lookup_id"],
            "selection_id":packet["selection_id"], "source_language":packet["source_language"],
            "target_language":packet["target_language"], "translation":"مرحبا"})
        bridge.publish(state)
    return {"accepted":True, "stop":packet["action"] == "stop"}
bridge = HeadlessTransport(sys.stdin.fileno(), sys.stdout, command, async_output=True)
bridge.publish(state)
while not bridge.poll(.01):
    pass
bridge.close()
''', encoding="utf-8")
            controller = WebController(core_path=child, python_executable=sys.executable,
                                       diagnostics_dir=directory / "logs", ack_timeout=2, stop_timeout=2)
            try:
                with TestClient(create_app(controller), base_url="http://127.0.0.1:8765", client=("127.0.0.1", 50000)) as client:
                    self.assertEqual(client.post("/api/start", json={}).status_code, 200)
                    body = {"text":"שלום", "source_language":"HE", "target_language":"AR", "selection_id":"one"}
                    response = client.post("/api/selection-translation", json=body)
                    self.assertEqual(response.status_code, 200, response.text)
                    identity = response.json()["lookup_id"]
                    deadline = time.monotonic() + 3
                    while True:
                        result = client.get("/api/selection-translation/" + identity).json()
                        if result["status"] != "pending" or time.monotonic() >= deadline:
                            break
                        time.sleep(.01)
                    self.assertEqual(result["status"], "complete", result)
                    self.assertEqual(result["translation"], "مرحبا")
                    self.assertEqual(result["selection_id"], "one")
                    cached = client.post("/api/selection-translation", json={**body, "selection_id":"two"}).json()
                    self.assertEqual(cached["status"], "complete")
                    self.assertEqual(cached["selection_id"], "two")
                    state = client.get("/api/state").json()
                    self.assertEqual(state["transcript"], [])
                    self.assertEqual(state["replies"], [])
                    self.assertNotIn("مرحبا", str(state))
                    self.assertEqual(client.post("/api/stop").status_code, 200)
            finally:
                controller.close()
