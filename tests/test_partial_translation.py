"""Confirmed-prefix batching without model inference, capture, or wall waits."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import pickle
from queue import Queue
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from tts_runtime.partial_translation import update_candidate
from tts_runtime.worker import ProcessStage


ROOT = Path(__file__).resolve().parents[1]


def update(previous, text, now, *, batch=True):
    return update_candidate(previous, text=text, locked_prefix=text, revision=int(now) + 1,
                            stable_text=text, now=now, batch_stable_prefix=batch)


def load_core():
    name = "partial_mt_core_" + uuid4().hex
    spec = importlib.util.spec_from_file_location(name, ROOT / "tts-transcribe.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules[name] = core
    with patch.object(sys, "argv", [str(spec.origin), "--source-language", "TR", "--target-language", "EN"]):
        spec.loader.exec_module(core)
    return core


class PartialTranslationPolicyTests(unittest.TestCase):
    def test_continuously_growing_confirmed_words_keep_the_oldest_unsent_deadline(self):
        candidate = None
        words = "alpha bravo charlie delta echo foxtrot".split()
        for now in range(3):
            candidate = update(candidate, " ".join(words[:now + 3]), now)
        self.assertEqual(candidate.stable_changed_at, 0)
        self.assertTrue(candidate.ready(2, 2))

    def test_correction_or_retraction_restarts_stabilization_on_token_boundaries(self):
        for revised in ["we need six chairs", "we need", "we need fivefold chairs"]:
            with self.subTest(revised=revised):
                candidate = update(None, "we need five", 0)
                candidate = update(candidate, revised, 1.5)
                self.assertEqual(candidate.stable_changed_at, 1.5)
                self.assertFalse(candidate.ready(2, 2))

    def test_successful_admissions_rate_limit_new_batches_and_new_words_get_an_age(self):
        candidate = update(None, "one two three", 0)
        candidate.admitted(candidate.stable_text, 2)
        candidate = update(candidate, "one two three four", 2.25)
        candidate = update(candidate, "one two three four five", 3)
        self.assertEqual(candidate.stable_changed_at, 2.25)
        self.assertFalse(candidate.ready(4, 2))
        self.assertTrue(candidate.ready(4.25, 2))
        candidate.admitted(candidate.stable_text, 4.25)
        self.assertFalse(candidate.ready(5, 2))

    def test_speculative_text_keeps_its_existing_quiet_period(self):
        candidate = update(None, "one two three", 0, batch=False)
        candidate = update(candidate, "one two three four", 1, batch=False)
        self.assertEqual(candidate.stable_changed_at, 1)
        self.assertFalse(candidate.ready(2, 2))

    def test_empty_prefix_does_not_start_age_for_words_that_do_not_exist_yet(self):
        candidate = update(None, "", 0)
        candidate = update(candidate, "one two three", 5)
        self.assertEqual(candidate.stable_changed_at, 5)
        self.assertFalse(candidate.ready(6, 2))

    def test_growth_pacing_preserves_first_preview_and_short_prefix_latency(self):
        candidate = update(None, "one two three", 0)
        self.assertFalse(candidate.needs_refresh(candidate.stable_text, 1.9, min_delta_words=2, interval=2))
        self.assertTrue(candidate.needs_refresh(candidate.stable_text, 2, min_delta_words=2, interval=2))
        candidate.admitted(candidate.stable_text, 2)
        candidate = update(candidate, "one two three four five", 2.25)
        self.assertFalse(candidate.needs_refresh(candidate.stable_text, 4, min_delta_words=2, interval=2))
        self.assertTrue(candidate.needs_refresh(candidate.stable_text, 4.25, min_delta_words=2, interval=2))

    def test_long_prefix_waits_for_proportionate_growth_before_retranslating(self):
        prefix = " ".join(f"word{i}" for i in range(40))
        candidate = update(None, prefix, 0)
        candidate.admitted(prefix, 2)
        candidate = update(candidate, prefix + " new words", 2.25)
        self.assertFalse(candidate.needs_refresh(candidate.stable_text, 4.25, min_delta_words=2, interval=2))
        growth = " ".join(f"new{i}" for i in range(8))
        candidate = update(candidate, prefix + " new words " + growth, 4.5)
        self.assertTrue(candidate.needs_refresh(candidate.stable_text, 4.5, min_delta_words=2, interval=2))

    def test_small_confirmed_growth_has_a_bounded_refresh_and_rejection_is_retryable(self):
        prefix = " ".join(f"word{i}" for i in range(40))
        candidate = update(None, prefix, 0)
        candidate.admitted(prefix, 2)
        candidate = update(candidate, prefix + " new words", 2.1)
        self.assertFalse(candidate.needs_refresh(candidate.stable_text, 7.99, min_delta_words=2, interval=2))
        self.assertTrue(candidate.needs_refresh(candidate.stable_text, 8, min_delta_words=2, interval=2))
        # A rejected queue admission does not call admitted(); the same words
        # remain eligible without losing the six-second refresh deadline.
        self.assertTrue(candidate.needs_refresh(candidate.stable_text, 8.1, min_delta_words=2, interval=2))
        self.assertEqual(candidate.last_sent_text, prefix)
        candidate.admitted(candidate.stable_text, 8.1)
        self.assertFalse(candidate.needs_refresh(candidate.stable_text, 20, min_delta_words=2, interval=2))

    def test_bounded_refresh_never_overrides_configured_word_or_interval_minimum(self):
        prefix = " ".join(f"word{i}" for i in range(40))
        candidate = update(None, prefix, 0)
        candidate.admitted(prefix, 10)
        candidate = update(candidate, prefix + " one", 10.1)
        self.assertFalse(candidate.needs_refresh(candidate.stable_text, 30, min_delta_words=2, interval=2))
        candidate = update(candidate, prefix + " one two", 10.2)
        self.assertFalse(candidate.needs_refresh(candidate.stable_text, 18, min_delta_words=2, interval=10))
        self.assertTrue(candidate.needs_refresh(candidate.stable_text, 20.2, min_delta_words=2, interval=10))

    def test_corrected_prefix_restarts_stabilization_without_waiting_for_growth(self):
        prefix = " ".join(f"word{i}" for i in range(40))
        candidate = update(None, prefix, 0)
        candidate.admitted(prefix, 2)
        corrected = "corrected " + prefix
        candidate = update(candidate, corrected, 4)
        self.assertEqual(candidate.last_sent_text, "")
        self.assertFalse(candidate.needs_refresh(corrected, 5.9, min_delta_words=2, interval=2))
        self.assertTrue(candidate.needs_refresh(corrected, 6, min_delta_words=2, interval=2))


class PartialTranslationCoordinatorTests(unittest.TestCase):
    def run_timeline(self, *, reject_first=False, busy=None, draining_after=None, min_delta_words=2,
                     asr_busy=None, pending_asr=None, word_counts=None):
        core = load_core()
        clock, snapshots, calls, errors = [0.0], [], [], []
        stop = threading.Event()
        ui = deepcopy(core.ui_state)
        ui.update(left_active={}, left_order=[], left_done=[])
        words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar".split()
        word_counts = list(word_counts if word_counts is not None else range(7, 15))
        words.extend(f"token{i}" for i in range(max(0, max(word_counts) - len(words))))

        class Stage:
            state = "ready"
            @property
            def snapshot(self):
                return {"state": "ready", **((busy or {}).get(int(clock[0]), {}))}
            def submit(self, task, *, partial=False, key=None):
                accepted = not (reject_first and partial and not calls)
                calls.append((clock[0], deepcopy(task), accepted))
                return accepted

        class AsrStage:
            @property
            def snapshot(self):
                return {"state": "ready", **((asr_busy or {}).get(int(clock[0]), {}))}

        class TimedQueue(Queue):
            def get(self, **_kwargs):
                now, result = Queue.get(self, block=False)
                clock[0] = now
                with core.final_asr_pending_lock:
                    core.final_asr_pending_utterances.clear()
                    core.final_asr_pending_utterances.update((pending_asr or {}).get(now, []))
                if draining_after is not None and now >= draining_after:
                    core.session_draining_event.set()
                return result
            def task_done(self):
                super().task_done()
                snapshots.append(deepcopy(ui))
                if self.unfinished_tasks == 0:
                    stop.set()

        results = TimedQueue()
        for now, count in enumerate(word_counts):
            request = {"session_id": "s", "source_language": "TR", "sample_rate": core.SAMPLE_RATE,
                       "channels": 1, "samples": (now + 1) * core.SAMPLE_RATE,
                       "timestamp_start_ns": 0, "timestamp_end_ns": (now + 1) * 1_000_000_000}
            result = core.AsrResult(1, " ".join(words[:count]), True, session_id="s", diagnostics={"request": request})
            results.put_nowait((now, result))
        results.put_nowait((len(word_counts), core.AsrResult(1, "authoritative source for the final translation", False)))
        with patch.dict(core.__dict__, {
            "shutdown_event": stop, "asr_results": results, "ui_state": ui,
            "asr_stage": AsrStage() if asr_busy is not None else None,
            "translation_stage": Stage(), "render_dashboard": lambda: None,
            "ui_update": lambda **kw: errors.append(kw), "record_asr_decision": lambda *a: None,
            "forced_bos_for_target": lambda _target: 0, "schedule_final_transcription_log": lambda *a: None,
        }), patch.object(core.time, "perf_counter", side_effect=lambda: clock[0]), \
                patch.object(core.args, "stable_mutable_tail_words", 1), \
                patch.object(core.args, "partial_translation_stabilize_seconds", 2), \
                patch.object(core.args, "partial_translation_min_delta_words", min_delta_words), \
                patch.object(core.args, "latency_stats", False):
            core.coordinator_worker()
            final = calls[-1][1]
            core.handle_translation_result(core.TranslationResult(source="audio", text="authoritative final translation",
                target_lang="EN", utterance_id=1, partial=False, revision=final.revision))
            before = deepcopy(ui)
            old = next(task for _time, task, accepted in calls if task.partial and accepted)
            core.handle_translation_result(core.TranslationResult(source="audio", text="obsolete partial translation",
                target_lang="EN", utterance_id=1, partial=True, revision=old.revision))
            self.assertEqual(ui, before)
        self.assertEqual(errors, [])
        self.assertTrue(snapshots[-1]["left_active"][1]["src_final"])
        self.assertEqual(calls[-1][1].partial, False)
        return calls

    def test_continuous_prefix_translates_before_final_and_paces_admissions(self):
        calls = self.run_timeline()
        partials = [(at, task) for at, task, accepted in calls if task.partial and accepted]
        self.assertEqual([at for at, _task in partials], [3, 6])
        self.assertLess(partials[0][0], calls[-1][0])

    def test_rejected_preview_is_retryable_without_marking_words_sent(self):
        calls = self.run_timeline(reject_first=True)
        partials = [(at, accepted) for at, task, accepted in calls if task.partial]
        self.assertEqual(partials[:2], [(3, False), (4, True)])
        self.assertTrue(all(at >= 6 for at, accepted in partials[2:] if accepted))

    def test_busy_optional_or_committed_mailbox_preserves_candidate_for_later(self):
        calls = self.run_timeline(busy={3: {"inflight_partial": True}, 4: {"committed_pending": 1},
                                        5: {"partials_pending": 1}, 8: {"committed_pending": 1}})
        self.assertEqual([at for at, task, accepted in calls if task.partial and accepted], [6])
        self.assertEqual(calls[-1][0], 8)
        self.assertFalse(calls[-1][1].partial)

    def test_drain_stops_optional_previews_but_still_admits_authoritative_final(self):
        calls = self.run_timeline(draining_after=4)
        self.assertEqual([at for at, task, accepted in calls if task.partial and accepted], [3])
        self.assertEqual(calls[-1][0], 8)
        self.assertFalse(calls[-1][1].partial)

    def test_committed_asr_work_defers_optional_mt_until_final_recognition_finishes(self):
        calls = self.run_timeline(asr_busy={
            3: {"committed_pending": 1},
            4: {"inflight": 20, "inflight_partial": False},
            8: {"committed_pending": 1},
        })
        self.assertEqual([at for at, task, accepted in calls if task.partial and accepted], [5])
        self.assertEqual(calls[-1][0], 8)
        self.assertFalse(calls[-1][1].partial)

    def test_regular_partial_asr_does_not_starve_live_translation(self):
        calls = self.run_timeline(asr_busy={
            now: {"inflight": now + 10, "inflight_partial": True} for now in range(8)
        })
        self.assertEqual([at for at, task, accepted in calls if task.partial and accepted], [3, 6])

    def test_final_asr_pending_before_mailbox_dispatch_also_defers_optional_mt(self):
        calls = self.run_timeline(pending_asr={3: [9], 4: [9]})
        self.assertEqual([at for at, task, accepted in calls if task.partial and accepted], [5])
        self.assertFalse(calls[-1][1].partial)

    def test_long_growing_prefix_keeps_first_preview_and_reduces_retranslation(self):
        calls = self.run_timeline(word_counts=range(40, 52))
        partials = [(at, task) for at, task, accepted in calls if task.partial and accepted]
        self.assertEqual([at for at, _task in partials], [3, 9])
        self.assertTrue(partials[1][1].text.startswith(partials[0][1].text + " "))
        self.assertGreater(len(partials[1][1].text.split()), len(partials[0][1].text.split()))
        self.assertLess(len(partials[1][1].text.split()) - len(partials[0][1].text.split()),
                        len(partials[0][1].text.split()) * 0.25)
        self.assertEqual(calls[-1][0], 12)
        self.assertFalse(calls[-1][1].partial)

    def test_batching_still_requires_the_configured_number_of_new_confirmed_words(self):
        calls = self.run_timeline(min_delta_words=999)
        self.assertEqual([at for at, task, accepted in calls if task.partial and accepted], [3])

    def test_final_work_supersedes_queued_preview_and_runs_before_optional_work(self):
        stage = ProcessStage("translation", lambda: None, lambda *_a: None, capacity=4)
        stage._state = "ready"  # Exercise mailbox ordering without starting a model process.
        self.assertTrue(stage.submit("preview one", partial=True, key=("audio", 1)))
        self.assertTrue(stage.submit("other preview", partial=True, key=("audio", 2)))
        self.assertTrue(stage.submit("final one", key=("audio", 1)))
        self.assertTrue(stage.submit("typed reply", key=("text", 3)))
        self.assertEqual(stage.snapshot["committed_pending"], 2)
        self.assertEqual(stage.snapshot["partials_pending"], 1)
        stage._state, stage._input_q = "ready", Queue()
        stage._dispatch_one()
        job = pickle.loads(stage._input_q.get_nowait())
        self.assertEqual(job.payload, "final one")
        self.assertFalse(job.partial)
        self.assertFalse(stage.submit("late preview one", partial=True, key=("audio", 1)))
