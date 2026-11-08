"""Slow output workers must not block input or lose accepted committed work."""

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
from queue import Empty, Full, PriorityQueue, Queue
import runpy
import sys
import threading
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class NonblockingOutputQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "argv", [str(ROOT / "tts-transcribe.py"), "--dump-config"]), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cls.scope = runpy.run_path(str(ROOT / "tts-transcribe.py"), run_name="queue_pressure_tests")
        cls.runtime = cls.scope["queue_put_translation_task"].__globals__

    def setUp(self):
        self.warnings = []
        self.stop = threading.Event()
        self.runtime_patch = patch.dict(self.runtime, {
            "shutdown_event": self.stop,
            "ui_update": lambda **values: self.warnings.append(values.get("status", "")),
        })
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)

    def translation(self, number, *, partial=False, source="text"):
        return self.scope["TranslationTask"](
            source=source, text=str(number), src_lang="eng_Latn", beams=1,
            target_lang="TR", utterance_id=number, partial=partial,
        )

    def speech(self, number, *, source="text"):
        return self.scope["TtsTask"](text=str(number), wav_out="unused.wav", lang_hint="TR",
                                      source=source, utterance_id=number)

    def call_without_waiting_for_consumer(self, callback):
        result = []
        failure = []

        def run():
            try:
                result.append(callback())
            except BaseException as exc:
                failure.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=0.5)
        stalled = thread.is_alive()
        if stalled:
            self.stop.set()
            thread.join(timeout=1)
        self.assertFalse(stalled, "producer waited for a consumer to free queue capacity")
        self.assertEqual(failure, [])
        return result[0]

    def test_full_committed_translation_queue_rejects_promptly_without_evicting(self):
        queue = Queue(maxsize=2)
        originals = [self.translation(1), self.translation(2, source="audio")]
        for task in originals:
            queue.put_nowait(task)
        accepted = self.call_without_waiting_for_consumer(
            lambda: self.scope["queue_put_translation_task"](queue, self.translation(3), "mt"))
        self.assertFalse(accepted)
        self.assertEqual([queue.get_nowait(), queue.get_nowait()], originals)
        queue.task_done()
        queue.task_done()
        self.assertTrue(any("typed reply was not queued" in message for message in self.warnings))
        self.assertEqual(queue.unfinished_tasks, 0)

    def test_full_typed_speech_queue_rejects_promptly_without_evicting(self):
        queue = Queue(maxsize=1)
        original = self.speech(1)
        queue.put_nowait(original)
        accepted = self.call_without_waiting_for_consumer(
            lambda: self.scope["queue_put_tts_task"](queue, self.speech(2), "tts"))
        self.assertFalse(accepted)
        self.assertIs(queue.get_nowait(), original)
        queue.task_done()
        self.assertTrue(any("not queued for speech" in message for message in self.warnings))

    def test_atomic_optional_removal_never_drains_committed_items_and_keeps_heap_accounting(self):
        queue = PriorityQueue(maxsize=3)
        committed = self.translation(1)
        final = self.translation(2, source="audio")
        partial = self.translation(3, source="audio", partial=True)
        for task in [final, partial, committed]:
            queue.put_nowait(self.scope["wrap_translation_task"](task))
        with patch.object(queue, "get_nowait", side_effect=AssertionError("must not drain a live queue")):
            self.assertTrue(self.scope["drop_one_speculative_translation_task"](queue, "mt"))
        self.assertEqual(queue.unfinished_tasks, 2)
        remaining = [self.scope["unwrap_translation_task"](queue.get_nowait()) for _ in range(2)]
        self.assertEqual(remaining, [committed, final])
        queue.task_done()
        queue.task_done()
        self.assertEqual(queue.unfinished_tasks, 0)

        speech_queue = Queue(maxsize=2)
        typed = self.speech(4)
        speech_queue.put_nowait(self.speech(5, source="audio"))
        speech_queue.put_nowait(typed)
        with patch.object(speech_queue, "get_nowait", side_effect=AssertionError("must not drain a live queue")):
            self.assertTrue(self.scope["drop_one_stale_audio_tts_task"](speech_queue, "tts"))
        self.assertIs(speech_queue.get_nowait(), typed)
        speech_queue.task_done()
        self.assertEqual(speech_queue.unfinished_tasks, 0)

    def test_legacy_queue_never_drains_or_reinserts_under_pressure(self):
        class LegacyFullQueue:
            def put_nowait(self, item):
                raise Full

            def get_nowait(self):
                raise AssertionError("legacy queue cannot be safely inspected")

        queue = LegacyFullQueue()
        self.assertFalse(self.scope["drop_one_speculative_translation_task"](queue, "mt"))
        self.assertFalse(self.scope["drop_one_stale_audio_tts_task"](queue, "tts"))
        self.assertFalse(self.call_without_waiting_for_consumer(
            lambda: self.scope["queue_put_translation_task"](queue, self.translation(1), "mt")))
        self.assertFalse(self.call_without_waiting_for_consumer(
            lambda: self.scope["queue_put_tts_task"](queue, self.speech(1), "tts")))

    def test_concurrent_pressure_preserves_every_accepted_committed_task_exactly_once(self):
        for kind in ("translation", "speech"):
            with self.subTest(kind=kind):
                queue = PriorityQueue(maxsize=4) if kind == "translation" else Queue(maxsize=4)
                accepted = []
                consumed = []
                failures = []
                lock = threading.Lock()
                start = threading.Barrier(5)
                finished = threading.Event()

                def committed(task):
                    return task.source == "text" or (kind == "translation" and not task.partial)

                def producer(producer_id):
                    try:
                        start.wait(timeout=2)
                        for offset in range(80):
                            number = producer_id * 1000 + offset
                            if kind == "translation":
                                optional = offset % 4 == 0
                                task = self.translation(number, source="audio" if offset % 2 == 0 else "text", partial=optional)
                                ok = self.scope["queue_put_translation_task"](queue, task, "mt")
                            else:
                                task = self.speech(number, source="audio" if offset % 4 == 0 else "text")
                                ok = self.scope["queue_put_tts_task"](queue, task, "tts")
                            if ok and committed(task):
                                with lock:
                                    accepted.append(number)
                            if offset % 9 == 0:
                                time.sleep(0)
                    except BaseException as exc:
                        failures.append(exc)

                def consumer():
                    try:
                        start.wait(timeout=2)
                        while not finished.is_set() or not queue.empty():
                            try:
                                item = queue.get_nowait()
                            except Empty:
                                time.sleep(0.0002)
                                continue
                            task = self.scope["unwrap_translation_task"](item) if kind == "translation" else item
                            if committed(task):
                                consumed.append(task.utterance_id)
                            queue.task_done()
                    except BaseException as exc:
                        failures.append(exc)

                consumer_thread = threading.Thread(target=consumer, daemon=True)
                producers = [threading.Thread(target=producer, args=(number,), daemon=True) for number in range(4)]
                consumer_thread.start()
                for thread in producers:
                    thread.start()
                for thread in producers:
                    thread.join(timeout=3)
                    self.assertFalse(thread.is_alive())
                finished.set()
                consumer_thread.join(timeout=3)
                self.assertFalse(consumer_thread.is_alive())
                self.assertEqual(failures, [])
                self.assertTrue(accepted)
                self.assertEqual(Counter(consumed), Counter(accepted))
                self.assertEqual(queue.unfinished_tasks, 0)


if __name__ == "__main__":
    unittest.main()
