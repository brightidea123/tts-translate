"""Owned subprocess lifecycle and JSONL command behavior, without devices/models."""
from __future__ import annotations

import json
from pathlib import Path
from queue import Queue
import signal
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tts_runtime.web_controller import WebController


class Lines:
    def __init__(self):
        self.lines = Queue()

    def readline(self, limit=-1):
        return self.lines.get()

    def emit(self, value):
        self.lines.put(value)


class Commands:
    def __init__(self, process):
        self.process = process
        self.text = ""
        self.block = None
        self.entered = threading.Event()

    def write(self, text):
        self.text += text

    def flush(self):
        self.entered.set()
        if self.block is not None:
            self.block.wait(2)
        while "\n" in self.text:
            line, self.text = self.text.split("\n", 1)
            packet = json.loads(line)
            self.process.commands.append(packet)
            self.process.received.set()
            response = self.process.respond(packet)
            if response is not None:
                self.process.emit({"type": "ack", "request_id": packet["request_id"], **response})
            if packet["action"] == "stop" and self.process.stop_on_command:
                self.process.finish(0)


class FakeProcess:
    next_pid = 99000000

    def __init__(self, *, respond=None, stop_on_command=True):
        type(self).next_pid += 1
        self.pid = type(self).next_pid
        self.code = None
        self.exited = threading.Event()
        self.received = threading.Event()
        self.stdout, self.stderr = Lines(), Lines()
        self.stdin = Commands(self)
        self.commands = []
        self.respond = respond or (lambda packet: {"accepted": True, "message": "accepted"})
        self.stop_on_command = stop_on_command

    def emit(self, packet):
        self.stdout.emit(json.dumps(packet, ensure_ascii=False) + "\n")

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        if not self.exited.wait(timeout):
            raise subprocess.TimeoutExpired("fake-core", timeout)
        return self.code

    def finish(self, code):
        if not self.exited.is_set():
            self.code = code
            self.stdout.emit("")
            self.stderr.emit("")
            self.exited.set()


class WebControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.processes, self.launches = [], []
        self.next_options = {}

        def launch(command, **kwargs):
            process = FakeProcess(**self.next_options)
            self.next_options = {}
            self.processes.append(process)
            observed = dict(kwargs)
            observed["env"] = {key: value for key, value in kwargs.get("env", {}).items()
                               if key in {"PYTHONUNBUFFERED", "TTS_WEB_RUN_ID"}}
            self.launches.append((command, observed))
            return process

        self.popen = patch("tts_runtime.web_controller.subprocess.Popen", side_effect=launch)
        self.signals = patch("tts_runtime.web_controller.os.killpg")
        self.popen.start()
        self.killpg = self.signals.start()
        self.controller = WebController(core_path=Path(self.temporary.name) / "core.py",
                                        python_executable="/test/python", ack_timeout=0.25,
                                        stop_timeout=0.3, diagnostics_dir=Path(self.temporary.name) / "logs")

    def tearDown(self):
        for process in self.processes:
            process.finish(0)
            if process.stdin.block is not None:
                process.stdin.block.set()
        self.controller.close()
        self.popen.stop()
        self.signals.stop()
        self.temporary.cleanup()

    def test_start_uses_only_owned_headless_process_and_rejects_parallel_session(self):
        state = self.controller.start({"source_language": "ar", "target_language": "he",
                                       "input_device": "LitLink Audio Bridge", "debug": True})
        self.assertTrue(state["accepted"])
        command, options = self.launches[0]
        self.assertEqual(command[:2], ["/test/python", "-u"])
        self.assertEqual(command[command.index("--ui") + 1], "headless")
        self.assertIn("--input-device=LitLink Audio Bridge", command)
        self.assertIn("--debug", command)
        self.assertEqual(command[command.index("--audio-review") + 1], "auto")
        self.assertTrue(options["start_new_session"])
        self.assertNotIn("shell", options)
        self.assertTrue(state["running"])
        rejected = self.controller.start()
        self.assertFalse(rejected["accepted"])
        self.assertEqual(rejected["controller_run_id"], state["controller_run_id"])
        self.assertEqual(len(self.launches), 1)

    def test_initial_cli_configuration_seeds_controls_without_starting_audio(self):
        controller = WebController(core_path=Path(self.temporary.name) / "core.py",
                                   initial_config={"source": "AR", "target": "HE", "debug": True,
                                                   "input_device": "LitLink"},
                                   diagnostics_dir=Path(self.temporary.name) / "initial-logs")
        self.assertEqual(controller.state()["config"]["source_language"], "AR")
        self.assertTrue(controller.state()["config"]["debug"])
        self.assertFalse(controller.state()["running"])
        self.assertEqual(self.launches, [])
        try:
            self.assertTrue(controller.start()["accepted"])
            command = self.launches[0][0]
            self.assertEqual(command[command.index("--target-language") + 1], "HE")
            self.assertIn("--input-device=LitLink", command)
        finally:
            controller.close()

    def test_start_preserves_cli_recording_policy_and_extra_runtime_arguments(self):
        controller = WebController(core_path=Path(self.temporary.name) / "core.py",
                                   initial_config={"source_language": "AR", "target_language": "HE", "audio_review": "off"},
                                   core_arguments=["--asr-preprocessing", "legacy", "--recordings-dir", "/tmp/review folder"],
                                   diagnostics_dir=Path(self.temporary.name) / "forwarded-logs")
        try:
            self.assertTrue(controller.start({"debug": True})["accepted"])
            command = self.launches[0][0]
            self.assertEqual(command[command.index("--audio-review") + 1], "off")
            self.assertEqual(command[command.index("--asr-preprocessing") + 1], "legacy")
            self.assertIn("/tmp/review folder", command)
            self.assertIn("--debug", command)
            self.assertEqual(controller.state()["config"]["source_language"], "AR")
        finally:
            controller.close()

    def test_unicode_reply_ack_retains_rejected_draft_and_clears_only_accepted(self):
        self.next_options = {"respond": lambda packet: {"accepted": packet.get("text") != "שלום مرحباً", "message": "full"}}
        self.controller.start()
        rejected = self.controller.reply("שלום مرحباً")
        self.assertFalse(rejected["accepted"])
        self.assertEqual(self.controller.state()["draft"], "שלום مرحباً")
        accepted = self.controller.reply("  Привет — Grüß dich\nمرحبا  ")
        self.assertTrue(accepted["accepted"])
        self.assertEqual(self.controller.state()["draft"], "")
        self.assertEqual(self.processes[0].commands[-1]["text"], "  Привет — Grüß dich\nمرحبا  ")
        self.assertNotEqual(rejected["request_id"], accepted["request_id"])
        self.assertTrue(self.controller.flush()["accepted"])
        self.assertTrue(self.controller.mark(7, "נדרש לבדוק")["accepted"])
        self.assertEqual(self.processes[0].commands[-1]["utterance_id"], 7)
        self.assertEqual(self.processes[0].commands[-1]["reason"], "נדרש לבדוק")

    def test_restart_resets_state_and_old_session_packets_cannot_contaminate_it(self):
        first = self.controller.start({"source": "AR", "target": "HE"})
        old_run = self.controller._run
        self.controller._packet(old_run, {"type": "state", "state": {"transcript": ["old audio"], "session_id": "old"}})
        old_run.process.finish(0)
        second = self.controller.start({"source": "FR", "target": "RU"})
        self.assertTrue(second["accepted"])
        self.assertNotEqual(first["controller_run_id"], second["controller_run_id"])
        self.controller._packet(old_run, {"type": "state", "state": {"transcript": ["late old audio"], "running": True}})
        state = self.controller.state()
        self.assertEqual(state["config"]["source_language"], "FR")
        self.assertEqual(state["transcript"], [])
        self.assertEqual(state["replies"], [])
        self.assertNotIn("old audio", str(state))
        self.processes[-1].finish(0)
        self.controller._packet(self.controller._run, {"type": "state", "state": {"running": True}})
        self.assertFalse(self.controller.state()["running"])

    def test_sent_timeout_is_explicitly_uncertain_and_late_ack_cannot_clear_draft(self):
        self.next_options = {"respond": lambda packet: None}
        self.controller.start()
        response = self.controller.reply("still editable")
        self.assertFalse(response["accepted"])
        self.assertTrue(response["uncertain"])
        self.assertEqual(self.controller.state()["draft"], "still editable")
        self.controller._packet(self.controller._run, {"type": "ack", "request_id": response["request_id"], "accepted": True})
        self.assertEqual(self.controller.state()["draft"], "still editable")

    def test_a_command_that_times_out_before_write_is_never_sent_later(self):
        self.controller.start()
        process = self.processes[0]
        process.stdin.block = threading.Event()
        first_result = []
        first = threading.Thread(target=lambda: first_result.append(self.controller.reply("first")))
        first.start()
        self.assertTrue(process.stdin.entered.wait(1))
        second = self.controller.reply("second")
        self.assertFalse(second["accepted"])
        self.assertFalse(second["uncertain"])
        process.stdin.block.set()
        first.join(1)
        self.assertFalse(first.is_alive())
        self.assertTrue(self.controller.reply("third")["accepted"])
        self.assertEqual([packet["text"] for packet in process.commands], ["first", "third"])

    def test_core_exit_wakes_waiters_and_stale_old_ack_never_acknowledges_new_run(self):
        self.next_options = {"respond": lambda packet: None}
        self.controller.start()
        old_run = self.controller._run
        response = []
        request = threading.Thread(target=lambda: response.append(self.controller.reply("old pending")))
        request.start()
        self.assertTrue(old_run.process.received.wait(1))
        old_run.process.finish(2)
        request.join(1)
        self.assertFalse(request.is_alive())
        self.assertFalse(response[0]["accepted"])
        self.assertIn("ended", response[0]["message"])
        self.next_options = {"respond": lambda packet: None}
        self.controller.start()
        process = self.processes[-1]
        new_response = []
        pending = threading.Thread(target=lambda: new_response.append(self.controller.reply("new pending")))
        pending.start()
        self.assertTrue(process.received.wait(1))
        request_id = process.commands[-1]["request_id"]
        self.controller._packet(old_run, {"type": "ack", "request_id": request_id, "accepted": True})
        self.assertFalse(new_response)
        self.controller._packet(self.controller._run, {"type": "ack", "request_id": request_id, "accepted": False, "message": "not ready"})
        pending.join(1)
        self.assertEqual(new_response[0]["message"], "not ready")
        self.assertEqual(self.controller.state()["draft"], "new pending")

    def test_stop_requests_grace_then_signals_only_the_owned_process_group(self):
        self.next_options = {"stop_on_command": False}
        self.controller.start()
        process = self.processes[-1]

        def signal_group(pid, sig):
            self.assertEqual(pid, process.pid)
            if sig == signal.SIGINT:
                process.finish(-2)

        self.killpg.side_effect = signal_group
        response = self.controller.stop()
        self.assertTrue(response["accepted"])
        self.assertFalse(response["running"])
        self.assertEqual(process.commands[-1]["action"], "stop")
        self.assertTrue(any(call.args == (process.pid, signal.SIGINT) for call in self.killpg.call_args_list))

    def test_devices_are_lazy_input_only_and_enumeration_errors_are_visible(self):
        sd = SimpleNamespace(default=SimpleNamespace(device=[1, -1]), query_devices=lambda: [
            {"name": "Speakers", "max_input_channels": 0, "default_samplerate": 48000},
            {"name": "LitLink", "max_input_channels": 2, "default_samplerate": 48000},
            {"name": "Microphone", "max_input_channels": 1, "default_samplerate": 16000},
        ])
        with patch.dict(sys.modules, {"sounddevice": sd}):
            devices = self.controller.devices()
            self.assertEqual([device["id"] for device in devices], [1, 2])
            self.assertTrue(devices[0]["default"])
            sd.query_devices = lambda: (_ for _ in ()).throw(RuntimeError("unavailable"))
            self.assertEqual(self.controller.devices(), [])
            self.assertIn("unavailable", self.controller.state()["device_error"])
        self.assertEqual(self.launches, [])

    def test_bad_configuration_or_actions_fail_before_spawning_or_sending(self):
        for config in [{"source": "XX"}, {"source": "AR", "target": "AR"}, {"debug": "yes"},
                       {"input_device": True}, {"audio_review": []}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.controller.start(config)
        for text in ["", "   ", None, "x" * 10001]:
            with self.assertRaises(ValueError):
                self.controller.reply(text)
        with self.assertRaises(ValueError):
            self.controller.mark(True)
        self.assertEqual(self.launches, [])

    def test_selection_admission_never_waits_for_core_ack_and_does_not_change_live_state(self):
        self.next_options = {"respond": lambda _packet: None}
        self.controller.start()
        before = self.controller.state()
        with patch.object(self.controller, "_command", side_effect=AssertionError("lookup must not use synchronous command waiting")):
            accepted = self.controller.selection_translation({"text": "שלום", "source_language": "HE",
                                                              "target_language": "AR", "selection_id": "selection-one"})
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["status"], "pending")
        process, run = self.processes[-1], self.controller._run
        self.assertTrue(process.received.wait(1))
        command = process.commands[-1]
        self.assertEqual(command["action"], "lookup")
        self.assertEqual(command["text"], "שלום")
        self.assertEqual(command["lookup_id"], accepted["lookup_id"])
        self.assertEqual(self.controller.state(), before)
        self.controller._packet(run, {"type": "ack", "request_id": command["request_id"], "accepted": True})
        self.controller._packet(run, {"type": "lookup_result", "lookup_id": accepted["lookup_id"],
                                      "selection_id": "selection-one", "source_language": "HE",
                                      "target_language": "AR", "translation": "مرحباً"})
        result = self.controller.selection_translation_result(accepted["lookup_id"])
        self.assertEqual((result["status"], result["translation"]), ("complete", "مرحباً"))
        self.assertEqual(self.controller.state(), before)
        cached = self.controller.selection_translation({"text": "שלום", "source_language": "HE",
                                                        "target_language": "AR", "selection_id": "selection-two"})
        self.assertEqual(cached["status"], "complete")
        self.assertEqual(cached["selection_id"], "selection-two")
        self.assertEqual(len(process.commands), 1)

    def test_selection_rejection_timeout_and_restart_cannot_publish_stale_results(self):
        self.next_options = {"respond": lambda _packet: None}
        self.controller.start()
        old_run = self.controller._run
        first = self.controller.selection_translation({"text": "hello", "source_language": "EN",
                                                       "target_language": "HE", "selection_id": "first"})
        self.assertTrue(old_run.process.received.wait(1))
        command = old_run.process.commands[-1]
        self.controller._packet(old_run, {"type": "ack", "request_id": command["request_id"],
                                          "accepted": False, "message": "Live translation has priority"})
        self.assertIn("priority", self.controller.selection_translation_result(first["lookup_id"])["error"])
        second = self.controller.selection_translation({"text": "next", "source_language": "EN",
                                                        "target_language": "HE", "selection_id": "second"})
        for pending in old_run.pending.values():
            if pending.lookup_id == second["lookup_id"]:
                pending.created_at -= 1
        self.assertEqual(self.controller.selection_translation_result(second["lookup_id"])["status"], "error")
        self.assertFalse(any(item.lookup_id == second["lookup_id"] for item in old_run.pending.values()))
        old_run.process.finish(0)
        self.controller.start()
        self.assertIsNone(self.controller.selection_translation_result(first["lookup_id"]))
        self.controller._packet(old_run, {"type": "lookup_result", "lookup_id": first["lookup_id"],
                                          "selection_id": "first", "source_language": "EN", "target_language": "HE",
                                          "translation": "stale"})
        self.assertIsNone(self.controller.selection_translation_result(first["lookup_id"]))


class RealJsonlTransportTests(unittest.TestCase):
    def test_real_child_pipe_preserves_unicode_acknowledgments_and_run_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            script = directory / "protocol_core.py"
            script.write_text('''import json, os, sys
state = {"session_id": os.environ["TTS_WEB_RUN_ID"], "transcript": [], "replies": []}
def send(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)
send({"type": "state", "state": state})
for line in sys.stdin:
    command = json.loads(line)
    send({"type": "ack", "request_id": command["request_id"], "accepted": True, "message": "accepted"})
    if command["action"] == "stop":
        break
    if command["action"] == "reply":
        state["replies"].append(command["text"])
    send({"type": "state", "state": state})
''', encoding="utf-8")
            controller = WebController(core_path=script, ack_timeout=2, stop_timeout=3,
                                       diagnostics_dir=directory / "logs")
            received_reply = threading.Event()
            original_packet = controller._packet

            def packet(run, value):
                original_packet(run, value)
                if value.get("state", {}).get("replies"):
                    received_reply.set()

            controller._packet = packet
            try:
                first = controller.start({"source": "AR", "target": "HE"})
                self.assertTrue(first["accepted"])
                phrase = "שלום مرحبا Привет — exact UTF-8"
                self.assertTrue(controller.reply(phrase)["accepted"])
                self.assertTrue(received_reply.wait(2))
                self.assertEqual(controller.state()["replies"], [phrase])
                self.assertEqual(controller.state()["session_id"], first["controller_run_id"])
                self.assertTrue(controller.stop()["accepted"])
                self.assertFalse(controller.state()["running"])
                second = controller.start({"source": "DE", "target": "FR"})
                self.assertTrue(second["accepted"])
                self.assertNotEqual(first["controller_run_id"], second["controller_run_id"])
                self.assertEqual(second["replies"], [])
                self.assertEqual(second["config"]["target_language"], "FR")
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
