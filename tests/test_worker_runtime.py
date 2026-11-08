"""Exercise real spawned model workers without importing a model or audio API."""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
import unittest

from tts_runtime.worker import ProcessStage


def initialize_ok():
    pass


def initialize_failure():
    raise RuntimeError("fixture model is missing")


def initialize_wait_then_fail(gate):
    gate.wait(10)
    initialize_failure()


def initialize_wait(gate):
    gate.wait(10)


def initialize_sleep(seconds):
    time.sleep(seconds)


def echo_handler(payload, emit):
    emit(payload)


def command_handler(payload, emit):
    if payload == "crash":
        os._exit(17)
    if payload == "hang":
        time.sleep(10)
    elif payload == "emit-then-hang":
        emit("first")
        time.sleep(10)
    elif isinstance(payload, tuple) and payload[0] == "wait":
        time.sleep(payload[1])
        emit(payload[2])
    else:
        emit(payload)


def raise_handler(payload, emit):
    raise ValueError("fixture inference failed")


def unpicklable_handler(payload, emit):
    emit(lambda: None)


class ProcessStageTests(unittest.TestCase):
    def setUp(self):
        self.stages = []

    def tearDown(self):
        for stage in self.stages:
            stage.close(timeout=0.7)

    def stage(self, initializer=initialize_ok, handler=echo_handler, **kwargs):
        stage = ProcessStage("test", initializer, handler,
                             startup_timeout=kwargs.pop("startup_timeout", 4),
                             request_timeout=kwargs.pop("request_timeout", 3), **kwargs)
        self.stages.append(stage)
        stage.start()
        return stage

    def wait_for(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("Worker did not reach the expected state within the test budget")

    def test_slow_model_start_never_blocks_submit_or_exceeds_capacity(self):
        events, results = [], []
        gate = mp.get_context("spawn").Event()
        stage = self.stage(initialize_wait, initargs=(gate,), capacity=3,
                           on_event=events.append, on_result=results.append)
        began = time.monotonic()
        self.assertTrue(stage.submit("one"))
        self.assertTrue(stage.submit("two"))
        self.assertTrue(stage.submit("three"))
        self.assertFalse(stage.submit("four"))
        self.assertLess(time.monotonic() - began, 0.2)
        self.assertEqual(stage.pending_count, 3)
        gate.set()
        self.wait_for(lambda: len(results) == 3)
        self.assertEqual(results, ["one", "two", "three"])
        self.wait_for(lambda: any(e.kind == "rejected" for e in events))
        rejected = next(e for e in events if e.kind == "rejected")
        self.assertIn("full of committed work", rejected.detail)
        self.assertTrue(any(e.kind == "loading" for e in events))
        self.assertTrue(any(e.kind == "ready" for e in events))

    def test_finals_keep_fifo_priority_and_coalesce_or_evict_only_partials(self):
        gate = mp.get_context("spawn").Event()
        results = []
        stage = self.stage(initialize_wait, initargs=(gate,), capacity=4, on_result=results.append)
        self.assertTrue(stage.submit("old partial", partial=True, key="a"))
        self.assertTrue(stage.submit("latest partial", partial=True, key="a"))
        self.assertTrue(stage.submit("unrelated partial", partial=True, key="b"))
        self.assertTrue(stage.submit("first final", key="a"))
        self.assertTrue(stage.submit("second final", key="c"))
        self.assertTrue(stage.submit("third final", key="d"))
        self.assertTrue(stage.submit("fourth final", key="e"))
        self.assertFalse(stage.submit("fifth final", key="f"))
        self.assertFalse(stage.submit("late partial", partial=True, key="a"))
        self.assertEqual(stage.pending_count, 4)
        gate.set()
        self.wait_for(lambda: len(results) == 4)
        self.assertEqual(results, ["first final", "second final", "third final", "fourth final"])
        self.assertEqual(stage.snapshot["dropped_partials"], 3)

    def test_selection_lane_is_bounded_and_runs_after_live_work_without_parallel_model_calls(self):
        gate = mp.get_context("spawn").Event()
        results, events = [], []
        stage = self.stage(initialize_wait, initargs=(gate,), capacity=4, on_result=results.append, on_event=events.append)
        self.assertTrue(stage.submit("old lookup", background=True, key=("lookup", "old")))
        self.assertTrue(stage.submit("latest lookup", background=True, key=("lookup", "new")))
        self.assertEqual(stage.snapshot["background_pending"], 1)
        self.assertTrue(stage.submit("preview", partial=True, key="preview"))
        self.assertTrue(stage.submit("final one", key="one"))
        self.assertTrue(stage.submit("final two", key="two"))
        gate.set()
        self.wait_for(lambda: len(results) == 4)
        self.assertEqual(results, ["final one", "final two", "preview", "latest lookup"])
        self.wait_for(lambda: any(event.kind == "discarded" and event.key == ("lookup", "old") for event in events))
        self.assertGreater(stage.snapshot["last_request_seconds"], 0)

    def test_full_live_mailbox_rejects_background_work_and_live_input_evicts_only_background(self):
        gate = mp.get_context("spawn").Event()
        results = []
        stage = self.stage(initialize_wait, initargs=(gate,), capacity=1, on_result=results.append)
        self.assertTrue(stage.submit("lookup", background=True, key=("lookup", 1)))
        self.assertTrue(stage.submit("live final", key="audio"))
        self.assertFalse(stage.submit("another lookup", background=True, key=("lookup", 2)))
        self.assertEqual(stage.pending_count, 1)
        gate.set(); self.wait_for(lambda: results == ["live final"])

    def test_superseded_inflight_partial_never_republishes_after_final_is_queued(self):
        results, events = [], []
        stage = self.stage(handler=command_handler, on_result=results.append, on_event=events.append)
        self.wait_for(lambda: stage.state == "ready")
        self.assertTrue(stage.submit(("wait", 0.3, "obsolete partial"), partial=True, key=12))
        self.wait_for(lambda: stage.state == "busy")
        self.assertTrue(stage.submit("authoritative final", key=12))
        self.wait_for(lambda: results == ["authoritative final"])
        self.wait_for(lambda: any(e.kind == "discarded" and e.partial for e in events))
        self.assertEqual(stage.snapshot["completed_jobs"], 2)

    def test_failed_initializer_rejects_queued_and_future_jobs_with_original_error(self):
        gate = mp.get_context("spawn").Event()
        events, results = [], []
        stage = self.stage(initialize_wait_then_fail, initargs=(gate,),
                           on_event=events.append, on_result=results.append)
        self.assertTrue(stage.submit("queued before model load"))
        gate.set()
        self.wait_for(lambda: stage.state == "failed")
        self.assertIn("fixture model is missing", stage.error)
        began = time.monotonic()
        self.assertFalse(stage.submit("typed after failure"))
        self.assertLess(time.monotonic() - began, 0.1)
        self.assertEqual(results, [])
        self.assertEqual(stage.pending_count, 0)
        self.wait_for(lambda: len([e for e in events if e.kind == "rejected"]) == 2)

    def test_rejected_replacement_does_not_suppress_the_current_partial_result(self):
        results = []
        stage = self.stage(handler=command_handler, capacity=1, on_result=results.append)
        self.wait_for(lambda: stage.state == "ready")
        self.assertTrue(stage.submit(("wait", 0.25, "current partial"), partial=True, key=1))
        self.wait_for(lambda: stage.state == "busy")
        self.assertTrue(stage.submit("other committed job", key=2))
        self.assertFalse(stage.submit("rejected same-utterance final", key=1))
        self.wait_for(lambda: results == ["current partial", "other committed job"])

    def test_crashed_child_is_reported_and_never_leaves_accepting_dead_mailbox(self):
        stage = self.stage(handler=command_handler)
        self.wait_for(lambda: stage.state == "ready")
        self.assertTrue(stage.submit("crash"))
        self.wait_for(lambda: stage.state == "failed")
        self.assertIn("exit code 17", stage.error)
        self.assertFalse(stage.submit("after crash"))

    def test_startup_and_request_watchdogs_stop_stalled_children(self):
        startup = self.stage(initialize_sleep, initargs=(10,), startup_timeout=0.25)
        self.assertTrue(startup.submit("waiting"))
        self.wait_for(lambda: startup.state == "failed", timeout=2)
        self.assertIn("initialization exceeded", startup.error)
        request = self.stage(handler=command_handler, request_timeout=0.2)
        self.wait_for(lambda: request.state == "ready")
        self.assertTrue(request.submit("hang"))
        self.wait_for(lambda: request.state == "failed", timeout=2)
        self.assertIn("inference exceeded", request.error)
        self.assertFalse(request.submit("later"))

    def test_slow_parent_callback_does_not_hide_native_worker_timeout(self):
        callback_entered, release_callback = threading.Event(), threading.Event()

        def blocked_callback(value):
            callback_entered.set()
            release_callback.wait(5)

        stage = self.stage(handler=command_handler, request_timeout=0.2, on_result=blocked_callback)
        self.wait_for(lambda: stage.state == "ready")
        self.assertTrue(stage.submit("emit-then-hang"))
        self.assertTrue(callback_entered.wait(2))
        try:
            self.wait_for(lambda: stage.state == "failed", timeout=2)
            self.assertIn("inference exceeded", stage.error)
            began = time.monotonic()
            self.assertFalse(stage.submit("after timeout"))
            stage.close(timeout=0.3)
            self.assertLess(time.monotonic() - began, 0.5)
        finally:
            release_callback.set()

    def test_handler_failure_is_terminal_and_unpicklable_output_is_visible(self):
        for handler, expected in [(raise_handler, "fixture inference failed"),
                                  (unpicklable_handler, "pickle")]:
            with self.subTest(handler=handler.__name__):
                stage = self.stage(handler=handler)
                self.wait_for(lambda: stage.state == "ready")
                self.assertTrue(stage.submit("input"))
                self.wait_for(lambda: stage.state == "failed")
                self.assertIn(expected, stage.error)
                self.assertFalse(stage.submit("retry in poisoned process"))

    def test_unpicklable_input_fails_in_dispatcher_without_blocking_submit(self):
        stage = self.stage()
        self.wait_for(lambda: stage.state == "ready")
        began = time.monotonic()
        self.assertTrue(stage.submit(lambda: None))
        self.assertLess(time.monotonic() - began, 0.1)
        self.wait_for(lambda: stage.state == "failed")
        self.assertIn("could not send inference job", stage.error)

    def test_close_cancels_loading_and_pending_work_with_bounded_wait(self):
        stage = self.stage(initialize_sleep, initargs=(10,))
        self.assertTrue(stage.submit("queued"))
        began = time.monotonic()
        stage.close(timeout=0.5)
        self.assertLess(time.monotonic() - began, 0.8)
        self.assertEqual(stage.state, "closed")
        self.assertEqual(stage.pending_count, 0)
        self.assertFalse(stage.submit("after close"))


if __name__ == "__main__":
    unittest.main()
