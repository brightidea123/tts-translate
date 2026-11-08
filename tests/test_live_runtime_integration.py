"""Live orchestration boundaries with real capture/endpoint code and fake models.

These tests never open devices, alter terminal state, or import model runtimes.
Synchronization uses events and ordered observations rather than latency limits.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_core():
    spec = importlib.util.spec_from_file_location("live_integration_core", ROOT / "tts-transcribe.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    with patch.object(sys, "argv", [str(spec.origin), "--source-language", "AR", "--target-language", "HE"]):
        spec.loader.exec_module(core)
    return core


class FakeStage:
    """A model stage that can remain loading forever without blocking submit."""

    def __init__(self, name, initializer=None, handler=None, *, accept=True, state="loading", **kwargs):
        self.name, self.initializer, self.handler = name, initializer, handler
        self.state, self.accept = state, accept
        self.on_event = kwargs.get("on_event")
        self.requests = []
        self.starts = 0
        self.closed = False

    @property
    def snapshot(self):
        return {"state": self.state, "error": "", "pending": len(self.requests), "inflight_key": None}

    def start(self):
        self.starts += 1
        if self.on_event:
            from tts_runtime.worker import StageEvent
            self.on_event(StageEvent("loading", self.name, "loading", "fake model still loading"))

    def submit(self, payload, *, partial=False, key=None):
        self.requests.append((payload, partial, key))
        return self.accept

    def close(self, timeout=2):
        self.closed = True
        self.state = "closed"


class LiveRuntimeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.core = load_core()

    def test_main_captures_and_accepts_typing_while_both_models_stay_loading(self):
        core = self.core
        order, stages, capture = [], {}, {}
        processed = threading.Event()
        original_on_frame = core.RingEndpointDetector.on_frame

        def on_frame(endpoint, timestamp_ns, pcm):
            original_on_frame(endpoint, timestamp_ns, pcm)
            capture["processed"] = pcm.copy()
            processed.set()

        def make_stage(name, initializer, handler, **kwargs):
            self.assertIn("terminal", order)
            stage = FakeStage(name, initializer, handler, **kwargs)
            stages[name] = stage
            order.append(name + " created")
            return stage

        class InputStream:
            def __init__(self, **kwargs):
                capture["kwargs"] = kwargs
                self.callback = kwargs["callback"]
                capture["feeder"] = self.callback.__self__

            def __enter__(stream):
                order.append("capture entered")
                self.assertEqual(stages["asr"].state, "loading")
                self.assertEqual(stages["translation"].state, "loading")
                stream.callback(np.full((core.FRAME_SAMPLES, 1), 2000, dtype=np.int16),
                                core.FRAME_SAMPLES, None, None)
                self.assertTrue(processed.wait(timeout=3), "endpoint never received PCM while models were loading")
                return stream

            def __exit__(self, *_args):
                order.append("capture closed")

        keys = iter([*"שלום", "\n", "\x1b"])
        forbidden = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("model operation ran on main/UI startup"))
        no_worker = lambda: None
        with patch.dict(core.__dict__, {
            "ProcessStage": make_stage, "sd": SimpleNamespace(InputStream=InputStream),
            "setup_terminal": lambda: order.append("terminal"), "restore_terminal": no_worker,
            "init_dashboard": lambda _lines: order.append("dashboard"), "render_dashboard": no_worker,
            "read_key": lambda _timeout: next(keys), "coordinator_worker": no_worker,
            "translation_result_worker": no_worker, "tts_worker": no_worker,
            "init_models": forbidden, "init_stt_runtime_only": forbidden,
            "warmup_stt_runtime_only": forbidden, "warmup_models": forbidden,
            "initialize_session_runtime": no_worker,
        }), patch.object(core.RingEndpointDetector, "on_frame", on_frame), redirect_stdout(io.StringIO()):
            self.assertEqual(core.main(), 0)
        self.assertEqual(capture["feeder"].captured_samples, core.FRAME_SAMPLES)
        self.assertTrue(np.array_equal(capture["processed"], np.full(core.FRAME_SAMPLES, 2000, dtype=np.int16)))
        self.assertEqual(capture["kwargs"]["samplerate"], 16000)
        self.assertLess(order.index("dashboard"), order.index("capture entered"))
        self.assertEqual(stages["asr"].starts, 1)
        self.assertEqual(stages["translation"].starts, 1)
        self.assertTrue(all(stage.closed for stage in stages.values()))
        self.assertEqual(len(stages["translation"].requests), 1)
        task = stages["translation"].requests[0][0]
        self.assertEqual((task.text, task.src_lang, task.target_lang), ("שלום", "heb_Hebr", "AR"))

    def test_rejected_typed_reply_remains_editable_without_a_pending_row_or_log(self):
        core = self.core
        for state in ("busy", "failed"):
            with self.subTest(state=state):
                stage = FakeStage("translation", state=state, accept=False)
                ui = deepcopy(core.ui_state)
                ui.update(right_active={}, right_order=[], right_done=[])
                rendered = []
                with patch.dict(core.__dict__, {"translation_stage": stage, "ui_state": ui,
                                               "typed_input_texts": {}, "render_dashboard": lambda: None,
                                               "render_input": lambda text, **_kw: rendered.append(text)}), \
                        patch.object(core, "append_session_log") as log:
                    self.assertFalse(core.submit_typed_reply("  שלום  "))
                    self.assertEqual(rendered[-1], "  שלום  ")
                    self.assertEqual(core.typed_input_texts, {})
                    self.assertEqual(ui["right_active"], {})
                    self.assertEqual(ui["right_order"], [])
                    self.assertEqual(ui["right_done"], [])
                    log.assert_not_called()
                    self.assertEqual(stage.requests[0][0].text, "שלום")

    def test_accepted_typed_reply_uses_reverse_language_direction(self):
        core = self.core
        stage = FakeStage("translation", accept=True)
        rendered = []
        with patch.dict(core.__dict__, {"translation_stage": stage, "render_dashboard": lambda: None,
                                       "render_input": lambda text, **_kw: rendered.append(text)}):
            self.assertTrue(core.submit_typed_reply("שלום"))
        task, partial, key = stage.requests[0]
        self.assertFalse(partial)
        self.assertEqual(key, ("text", task.utterance_id))
        self.assertEqual((task.src_lang, task.target_lang), ("heb_Hebr", "AR"))
        self.assertEqual(task.forced_bos_id, core.forced_bos_for_target("AR"))
        self.assertEqual(rendered, [""])

    def test_full_asr_stage_rejects_final_without_stopping_later_capture(self):
        core = self.core
        stage = FakeStage("asr", state="busy", accept=False)
        done = threading.Event()
        ring = core.NativeTimestampRingBuffer(10)
        scheduler = core.PriorityAsrScheduler()
        endpoint = core.RingEndpointDetector(ring, scheduler)
        endpoint.vad_gate = SimpleNamespace(detect=lambda _pcm: (True, -20.0, 1.0))
        original = endpoint.on_frame
        seen = []

        def consume(timestamp_ns, pcm):
            original(timestamp_ns, pcm)
            seen.append(pcm.copy())
            if len(seen) == 110:
                done.set()

        endpoint.on_frame = consume
        with patch.dict(core.__dict__, {"asr_stage": stage, "ui_update": lambda **_kw: None,
                                       "render_dashboard": lambda: None}), \
                patch.object(core.args, "streaming_asr_chunk_seconds", 100.0):
            # This direct forced-final call must return on rejection, without
            # entering the old blocking priority queue fallback.
            endpoint.in_speech = True
            endpoint.utterance_id = 77
            endpoint.active = [np.full(core.FRAME_SAMPLES, 1, dtype=np.int16)]
            endpoint.speech_frames = endpoint.total_frames = 1
            endpoint.finalize(force=True)
            self.assertFalse(core.is_final_asr_pending(77))
            self.assertNotIn(77, core.pending_stage_asr)
            self.assertNotIn(77, scheduler.final_submitted)
            with core.RingAudioFeeder(endpoint, queue_ms=2500) as feeder:
                for value in range(110):
                    feeder.callback(np.full((core.FRAME_SAMPLES, 1), value, dtype=np.int16),
                                    core.FRAME_SAMPLES, None, None)
                self.assertTrue(done.wait(timeout=3), "capture/endpoint stopped behind a failed ASR submission")
            self.assertEqual(feeder.captured_samples, 110 * core.FRAME_SAMPLES)
            self.assertEqual(feeder.dropped_samples, 0)
        self.assertTrue(np.array_equal(np.concatenate(seen), np.repeat(np.arange(110, dtype=np.int16), core.FRAME_SAMPLES)))
        self.assertTrue(np.array_equal(ring.latest_ms(2200), np.concatenate(seen)))
        self.assertTrue(all(not partial for _payload, partial, _key in stage.requests))

    def test_asr_final_is_displayed_even_when_translation_stage_has_failed(self):
        core = self.core
        stop = threading.Event()

        class FiniteResults(Queue):
            def task_done(self):
                super().task_done()
                stop.set()

        results = FiniteResults()
        text = "مرحبا هذا اختبار للبث المباشر"
        result = core.AsrResult(utterance_id=101, text=text, partial=False)
        stage = FakeStage("translation", state="failed", accept=False)
        ui = deepcopy(core.ui_state)
        ui.update(left_active={}, left_order=[], left_done=[])
        with patch.dict(core.__dict__, {"shutdown_event": stop, "asr_results": results,
                                       "translation_stage": stage, "ui_state": ui,
                                       "audio_final_transcripts": {}, "render_dashboard": lambda: None,
                                       "prepare_final_transcript_for_translation": lambda current, _previous: current}):
            core.receive_asr_stage_result(result)
            core.coordinator_worker()
            self.assertIn(text, core.strip_ansi(ui["left_active"][101]["src"]))
            self.assertEqual(core.audio_final_transcripts[101], text)
            self.assertEqual(len(stage.requests), 1)
            self.assertEqual(stage.requests[0][0].source, "audio")
            self.assertFalse(core.is_final_asr_pending(101))
            self.assertEqual(results.unfinished_tasks, 0)

    def test_failed_asr_stage_releases_pending_final_gates(self):
        core = self.core
        scheduler = core.PriorityAsrScheduler()
        scheduler.final_submitted.add(25)
        scheduler.cancelled_partials.add(25)
        core.mark_final_asr_pending(25)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(core.__dict__, {
            "__file__": str(Path(temporary) / "app.py"), "ring_scheduler": scheduler,
            "pending_stage_asr": {25}, "render_dashboard": lambda: None,
        }):
            core.on_inference_event(core.StageEvent("failed", "asr", "failed", "fake startup failure"))
            self.assertEqual(core.pending_stage_asr, set())
            self.assertFalse(core.is_final_asr_pending(25))
            self.assertNotIn(25, scheduler.final_submitted)
            self.assertNotIn(25, scheduler.cancelled_partials)

    def test_strict_asr_warmup_failure_does_not_advertise_ready(self):
        core = self.core
        with patch.dict(core.__dict__, {"detected_stt_backend": "mlx-whisper"}), \
                patch.object(core.args, "warmup", True), \
                patch.object(core, "configure_inference_diagnostics"), \
                patch.object(core, "init_stt_runtime_only"), \
                patch.object(core, "ensure_mlx_whisper_loaded"), \
                patch.object(core, "transcribe_audio_mlx_whisper", side_effect=RuntimeError("fake Metal initialization failure")):
            with self.assertRaisesRegex(RuntimeError, "ASR initialization/warmup failed.*fake Metal"):
                core.initialize_asr_stage()

    def test_completed_translation_replaces_only_optional_preview(self):
        core = self.core
        results = Queue(maxsize=1)
        preview = core.TranslationResult("audio", "draft", "HE", partial=True, stream_delta=True)
        final = core.TranslationResult("audio", "final", "HE", utterance_id=1)
        results.put_nowait(preview)
        with patch.dict(core.__dict__, {"translation_stage_results": results}):
            core.receive_translation_stage_result(final)
            self.assertIs(results.queue[0], final)
            self.assertEqual(results.unfinished_tasks, 1)
            # The reverse replacement must never discard an accepted final.
            core.receive_translation_stage_result(preview)
            self.assertIs(results.get_nowait(), final)
            results.task_done()
        self.assertEqual(results.unfinished_tasks, 0)

    def test_committed_translation_backpressure_waits_only_in_delivery_thread(self):
        core = self.core
        waiting = threading.Event()
        delivered = threading.Event()
        stop = threading.Event()

        class ObservedQueue(Queue):
            def put(self, item, block=True, timeout=None):
                if timeout is not None:
                    waiting.set()
                return super().put(item, block=block, timeout=timeout)

        results = ObservedQueue(maxsize=1)
        first = core.TranslationResult("text", "first accepted reply", "AR", utterance_id=1)
        second = core.TranslationResult("text", "second accepted reply", "AR", utterance_id=2)
        results.put_nowait(first)

        def deliver():
            core.receive_translation_stage_result(second)
            delivered.set()

        with patch.dict(core.__dict__, {"translation_stage_results": results, "shutdown_event": stop}):
            delivery = threading.Thread(target=deliver, daemon=True)
            delivery.start()
            try:
                self.assertTrue(waiting.wait(timeout=3), "completed output was discarded instead of awaiting delivery capacity")
                self.assertFalse(delivered.is_set())
                feeder = core.RingAudioFeeder(SimpleNamespace(), queue_ms=40)
                feeder.callback(np.ones((core.FRAME_SAMPLES, 1), dtype=np.int16), core.FRAME_SAMPLES, None, None)
                self.assertEqual(feeder.captured_samples, core.FRAME_SAMPLES)
                self.assertIs(results.get_nowait(), first)
                results.task_done()
                self.assertTrue(delivered.wait(timeout=3))
                delivery.join(timeout=3)
                self.assertFalse(delivery.is_alive())
                self.assertIs(results.get_nowait(), second)
                results.task_done()
                self.assertEqual(results.unfinished_tasks, 0)
            finally:
                stop.set()
                delivery.join(timeout=3)

    def test_watchdog_recovers_more_stuck_partials_than_its_own_result_queue_can_hold(self):
        core = self.core
        stop = threading.Event()
        recovered = threading.Event()
        results = Queue(maxsize=2)
        expected_ids = list(range(200, 207))

        class RecordingStage(FakeStage):
            def submit(self, payload, **kwargs):
                accepted = super().submit(payload, **kwargs)
                if len(self.requests) == len(expected_ids):
                    recovered.set()
                    stop.set()
                return accepted

        stage = RecordingStage("translation", accept=True)
        ui = deepcopy(core.ui_state)
        ui.update(left_active={}, left_order=[], left_done=[])
        with patch.dict(core.__dict__, {
            "shutdown_event": stop, "asr_results": results, "translation_stage": stage,
            "ui_state": ui, "ring_endpoint": None, "render_dashboard": lambda: None,
            "sanitize_asr_text": lambda text, **_kwargs: text,
            "prepare_final_transcript_for_translation": lambda current, _previous: current,
        }), patch.object(core.args, "stuck_partial_timeout_ms", 500), \
                patch.object(core.args, "stuck_partial_watchdog", True), \
                patch.object(core.args, "stuck_partial_action", "finalize"), \
                patch.object(core.args, "stream_audio_translation", False), \
                patch.object(core.args, "partial_translation_policy", "off"), \
                patch.object(core.args, "final_dedupe_window_ms", 0):
            worker = threading.Thread(target=core.coordinator_worker, daemon=True)
            worker.start()
            try:
                for utterance_id in expected_ids:
                    results.put(core.AsrResult(utterance_id, f"رسالة عربية مستقلة {utterance_id}", partial=True), timeout=2)
                self.assertTrue(recovered.wait(timeout=3), "watchdog blocked while producing finals into the queue it must also consume")
            finally:
                stop.set()
                worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertEqual([task.utterance_id for task, _partial, _key in stage.requests], expected_ids)
            self.assertTrue(all(not task.partial for task, _partial, _key in stage.requests))
            self.assertEqual(results.unfinished_tasks, 0)


if __name__ == "__main__":
    unittest.main()
