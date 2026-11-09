from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from queue import Empty
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_accuracy_integration import load_core
from tts_runtime.selection_inference import SelectionTask, SelectionResult


class SelectionInferenceTests(unittest.TestCase):
    def setUp(self):
        self.core = load_core()
        self.core.runtime_session_id = "session-current"
        self.packets = []
        self.core.headless_transport = Mock(send_packet=lambda packet: self.packets.append(packet))
        self.core.translation_stage = Mock(state="ready")
        self.core.translation_stage.submit.return_value = True
        self.command = {"action": "lookup", "lookup_id": "a" * 32, "selection_id": "selected",
                        "text": "שלום", "source_language": "HE", "target_language": "AR"}

    def test_admission_uses_optional_lane_and_return_path_never_enters_conversation_or_speech(self):
        core = self.core
        self.assertTrue(core.process_headless_action(self.command)["accepted"])
        task = core.translation_stage.submit.call_args.args[0]
        self.assertIsInstance(task, SelectionTask)
        self.assertEqual(core.translation_stage.submit.call_args.kwargs, {"background": True, "key": ("lookup", "a" * 32)})
        before = dict(core.ui_state)
        core.receive_translation_stage_result(SelectionResult(task, "مرحباً<end_of_turn>"))
        self.assertEqual(self.packets[0]["translation"], "مرحباً")
        self.assertEqual(self.packets[0]["selection_id"], "selected")
        self.assertEqual(core.ui_state, before)
        self.assertTrue(core.tts_tasks.empty())
        self.assertTrue(core.translation_stage_results.empty())
        self.assertEqual(core.selection_tasks, {})

    def test_child_serializes_selected_direction_through_existing_model_without_deltas(self):
        core = self.core
        task = SelectionTask.from_command(self.command, "session-current")
        emit = Mock()
        with patch.object(core, "translate_chunk", return_value="مرحباً") as translate, \
                patch.object(core, "forced_bos_for_target", return_value=8) as target, \
                patch.object(core, "emit_translation_results", side_effect=AssertionError("lookup must not use conversation output")):
            core.handle_translation_stage_request(task, emit)
        translate.assert_called_once_with("שלום", "heb_Hebr", 8, 1)
        target.assert_called_once_with("AR")
        self.assertEqual(emit.call_args.args[0], SelectionResult(task, "مرحباً"))

    def test_foreign_results_and_rejections_cannot_complete_or_mutate_another_lookup(self):
        core = self.core
        core.submit_selection_task(self.command)
        task = core.selection_tasks["a" * 32]
        foreign = SelectionTask(task.lookup_id, task.selection_id, "old-session", task.text, "HE", "AR")
        core.publish_selection_result(SelectionResult(foreign, "wrong"))
        self.assertEqual(self.packets, [])
        core.on_inference_event(core.StageEvent("discarded", "translation", "ready", "Live work has priority", key=("lookup", task.lookup_id), background=True))
        self.assertEqual(self.packets[-1]["error"], "Live work has priority")
        core.publish_selection_result(SelectionResult(task, "late"))
        self.assertEqual(len(self.packets), 1)

    def test_lookup_protocol_validates_identity_and_bounded_selection(self):
        for values in [{"lookup_id": "bad"}, {"target_language": "HE"}, {"text": "x" * 1501}]:
            with self.subTest(values=str(values)[:70]), self.assertRaises(ValueError):
                SelectionTask.from_command({**self.command, **values}, "session")

    def test_busy_preview_is_allowed_to_finish_and_final_priority_is_preserved(self):
        core = self.core
        core.asr_stage = Mock(state="busy")
        core.asr_stage.snapshot = {"inflight_partial": True, "state": "busy", "partials_pending": 0}
        core.asr_stage.submit.return_value = True
        scheduler = core.PriorityAsrScheduler()
        pcm = np.arange(16000, dtype=np.int16)
        self.assertFalse(scheduler.submit(3, pcm, partial=True))
        core.asr_stage.submit.assert_not_called()
        self.assertTrue(scheduler.submit(3, pcm, partial=False))
        self.assertFalse(core.asr_stage.submit.call_args.kwargs["partial"])

    def test_collapsed_final_without_recovery_input_is_unresolved_without_promoting_preview(self):
        core = self.core
        core.asr_request_context = {"utterance_id": 3}
        preview = "Önce kendimiz yaşayıp hem de ekranlarınızın başında sizler çocuklarınız aileleriniz"
        final = "İzlediğiniz için teşekkür ederim."
        def decode(_pcm, **kwargs):
            text = preview if kwargs["partial"] else final
            core.last_decoder_quality.update(raw_text=text, max_no_speech_prob=.1, max_compression_ratio=1)
            return text
        with patch.object(core, "_transcribe_pcm_i16_impl", side_effect=decode):
            core.transcribe_pcm_i16(np.full(96000, 4000, dtype=np.int16), partial=True)
            answer = core.transcribe_pcm_i16(np.full(298880, 4000, dtype=np.int16), partial=False)
        self.assertEqual(answer, "")
        self.assertIn("final_preview_disagreement", core.last_asr_diagnostics["issues"])
        self.assertIn("final_recovery_unresolved", core.last_asr_diagnostics["issues"])
        self.assertEqual(core.last_asr_diagnostics["preview_evidence"], preview)

    def test_decoder_loop_preview_is_review_only_while_authoritative_final_and_emphasis_survive(self):
        core = self.core
        repeated = "dünyanın en büyük ve en büyük ve en büyük ve en büyük"
        core.receive_asr_stage_result(core.AsrResult(4, repeated, True, diagnostics={"decoder": {"max_compression_ratio": 8.5}}))
        self.assertTrue(core.asr_results.empty())
        final = core.AsrResult(4, repeated, False, diagnostics={"decoder": {"max_compression_ratio": 8.5}})
        core.receive_asr_stage_result(final)
        self.assertIs(core.asr_results.get_nowait(), final)
        core.asr_results.task_done()
        emphasis = core.AsrResult(5, "very very very good", True, diagnostics={"decoder": {"max_compression_ratio": 1.5}})
        core.receive_asr_stage_result(emphasis)
        self.assertIs(core.asr_results.get_nowait(), emphasis)
        core.asr_results.task_done()
