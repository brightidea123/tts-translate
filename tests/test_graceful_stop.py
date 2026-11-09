"""Stop drains captured tails and final text work without models or devices."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
from queue import Queue
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_core():
    name = "stop_core_" + uuid4().hex
    spec = importlib.util.spec_from_file_location(name, ROOT / "tts-transcribe.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules[name] = core
    with patch.object(sys, "argv", [str(spec.origin), "--source-language", "HE", "--target-language", "AR", "--no-session-logs", "--audio-review", "off"]):
        spec.loader.exec_module(core)
    return core


class Stage:
    def __init__(self, state="ready", **values):
        self.values = {"state": state, "pending": 0, "committed_pending": 0,
                       "inflight": None, "pending_callbacks": 0, **values}

    @property
    def snapshot(self):
        return dict(self.values)


class GracefulStopTests(unittest.TestCase):
    def setUp(self):
        self.core = load_core()
        self.core.render_dashboard = lambda: None

    def test_headless_stop_or_eof_stops_admission_without_stopping_consumers(self):
        core = self.core
        core.headless_transport = SimpleNamespace(poll=lambda _timeout: True, snapshot={})
        self.assertEqual(core.read_headless_command(0), "\x1b")
        self.assertTrue(core.session_draining_event.is_set())
        self.assertFalse(core.shutdown_event.is_set())
        started = core.session_drain_started_at
        core.request_session_drain()
        self.assertEqual(core.session_drain_started_at, started)

    def test_actual_feeder_flushes_subframe_tail_and_waits_for_final_asr_then_mt_delivery(self):
        core = self.core
        frames, finals, translated = [], [], []
        class Endpoint:
            def on_frame(self, _timestamp, pcm):
                frames.append(pcm.copy())
            def finalize(self, force=False):
                self_test.assertFalse(core.shutdown_event.is_set())
                self_test.assertTrue(core.session_draining_event.is_set())
                finals.append(force)
                core.mark_final_asr_pending(1)
                core.asr_results.put(core.AsrResult(1, "מקור סופי", False))
        class TranslationStage(Stage):
            def submit(self, task, **_kwargs):
                translated.append(task)
                core.receive_translation_stage_result(core.TranslationResult(
                    source="audio", text="ترجمة نهائية", target_lang="AR", utterance_id=1,
                    partial=False, revision=task.revision, final=True))
                return True
        self_test = self
        core.asr_stage = Stage()
        core.translation_stage = TranslationStage()
        core.forced_bos_for_target = lambda _lang: None
        core.ui_set_panel_entry("left", 1, "src", "HE~: provisional", False,
                               preview=core.source_preview("provisional", "", inflight=False))
        consumers = [threading.Thread(target=core.coordinator_worker), threading.Thread(target=core.translation_result_worker)]
        for thread in consumers: thread.start()
        feeder = core.RingAudioFeeder(Endpoint()).__enter__()
        try:
            pcm = np.arange(480, dtype=np.int16)
            feeder.callback(pcm.reshape(-1, 1), len(pcm), None, None)
            report = core.drain_session_work(feeder, timeout=2)
            self.assertEqual(report["status"], "completed")
            self.assertFalse(core.shutdown_event.is_set())
            self.assertEqual(finals, [True])
            np.testing.assert_array_equal(np.concatenate(frames), pcm)
            self.assertEqual(len(translated), 1)
            self.assertEqual(translated[0].text, "מקור סופי")
            self.assertNotIn(1, core.ui_state["left_active"])
            self.assertIn("מקור סופי", core.strip_ansi(core.ui_state["left_done"][0]["src"]))
            self.assertIn("ترجمة نهائية", core.strip_ansi(core.ui_state["left_done"][0]["dst"]))
        finally:
            core.shutdown_event.set()
            feeder.close()
            for thread in consumers: thread.join(timeout=1)
        self.assertTrue(all(not thread.is_alive() for thread in consumers))

    def test_pending_callbacks_and_queue_consumer_work_are_part_of_drain(self):
        core = self.core
        core.asr_stage = Stage(pending_callbacks=1)
        core.asr_results.put("delivering")
        core.asr_results.get_nowait()
        snapshot = core.session_drain_snapshot()
        self.assertEqual(snapshot["queues"]["asr_results"], 1)
        self.assertEqual(snapshot["stages"]["asr"]["pending_callbacks"], 1)
        core.asr_results.task_done()
        core.asr_stage.values["pending_callbacks"] = 0
        self.assertEqual(core.drain_session_work(timeout=1)["status"], "completed")

    def test_loading_or_failed_models_do_not_spend_the_full_grace_period(self):
        core = self.core
        for state in ("loading", "failed"):
            with self.subTest(state=state):
                core.asr_stage = Stage(state, committed_pending=1)
                report = core.drain_session_work(timeout=6)
                self.assertEqual(report["status"], "unavailable")
                self.assertLess(report["elapsed_seconds"], 1)
                self.assertIn("asr", report["reason"])
                self.assertFalse(core.shutdown_event.is_set())

    def test_timeout_retains_real_partial_and_explicitly_records_incomplete_work(self):
        core = self.core
        with tempfile.TemporaryDirectory() as directory:
            core.runtime_session_dir = Path(directory)
            core.args.session_logs = True
            core.ui_set_panel_entry("left", 7, "src", "HE~: unfinished words", False,
                                   preview=core.source_preview("unfinished words", "", inflight=False))
            core.mark_final_asr_pending(7)
            report = core.drain_session_work(timeout=0)
            self.assertEqual(report["status"], "timeout")
            self.assertEqual(report["pending"]["pending_final_asr"], [7])
            self.assertFalse(core.ui_state["left_active"][7]["src_final"])
            self.assertIn("unfinished words", report["retained_text"][0]["source"])
            self.assertTrue((Path(directory) / "shutdown-drain.json").exists())
            self.assertFalse(core.shutdown_event.is_set())

    def test_final_debounce_log_is_flushed_before_stop_without_duplicate_timer_write(self):
        core = self.core
        core.pending_audio_translation_logs[2] = (3, "final source", "final translation")
        with patch.object(core, "log_final_transcription") as log:
            self.assertEqual(core.drain_session_work(timeout=1)["status"], "completed")
            core._flush_final_logs_on_stop()
            log.assert_called_once_with(2, "final source", "final translation")
        self.assertEqual(core.pending_audio_translation_logs, {})

    def test_drain_failure_cannot_bypass_main_worker_cleanup(self):
        core = self.core
        core.args.mode = "text"
        core.args.mt_process = False
        with patch.dict(core.__dict__, {
            "initialize_session_runtime": lambda: None, "init_parent_routing_runtime": lambda: None,
            "setup_terminal": lambda: None, "restore_terminal": lambda: None,
            "init_dashboard": lambda _lines: None, "translation_worker": lambda: None,
            "tts_worker": lambda: None, "read_key": lambda _timeout: "\x1b",
        }), patch.object(core, "drain_session_work", side_effect=OSError("disk full")), \
                patch.object(core, "stop_asr_process") as asr, patch.object(core, "stop_mt_process") as mt, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(core.main(), 0)
            asr.assert_called_once()
            mt.assert_called_once()
            self.assertTrue(core.shutdown_event.is_set())

    def test_web_controller_budget_covers_the_configured_drain(self):
        core = self.core
        core.args.shutdown_grace_seconds = 15
        with patch("tts_runtime.web_controller.WebController") as controller, \
                patch("tts_runtime.web_server.WebInterface") as server, \
                patch.object(core.time, "sleep", side_effect=KeyboardInterrupt), \
                patch.object(sys, "argv", ["tts-transcribe.py", "--ui", "web", "--shutdown-grace-seconds", "15"]), \
                redirect_stderr(io.StringIO()):
            self.assertEqual(core.run_web_interface(), 0)
            self.assertEqual(controller.call_args.kwargs["stop_timeout"], 19)
            self.assertIn("--shutdown-grace-seconds", controller.call_args.kwargs["core_arguments"])
            controller.return_value.close.assert_called_once()
            server.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
