"""Recorded-artifact checks run offline and never execute the live pipeline."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from scripts import audit_sessions


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_wav(path, pcm):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class SessionAuditTests(unittest.TestCase):
    def fixture(self, directory):
        pcm = np.array([0, 32767, -32768, 1234, 5000, -6000, 111, 222], dtype="<i2").tobytes()
        write_wav(directory / "capture-stream.wav", pcm)
        write_json(directory / "capture-stream.json", {"kind": "stream_segment", "audio_file": "capture-stream.wav",
                   "timestamp_ns": 1_000_000_000, "sample_count": 8, "pcm_sha256": digest(pcm),
                   "file_sha256": digest((directory / "capture-stream.wav").read_bytes())})
        return pcm

    def add_request(self, directory, pcm, uid, offset, *, partial=False, raw_text="שלום", text="שלום", with_input=True):
        request_id = f"u{uid}-{'p' if partial else 'f'}"
        clip = pcm[offset * 2:(offset + 2) * 2]
        name = request_id + "-raw.wav"
        write_wav(directory / name, clip)
        request = {"request_id": request_id, "utterance_id": uid, "pcm_sha256": digest(clip),
                   "actual_pcm_sha256": digest(clip), "timestamp_start_ns": 1_000_000_000 + offset * 62_500,
                   "timestamp_end_ns": 1_000_000_000 + (offset + 2) * 62_500, "source_span_minus_pcm_ns": 0}
        prepared = (np.frombuffer(clip, dtype="<i2").astype(np.float32) / 32768).astype("<f4").tobytes()
        metadata = {"request": request, "partial": partial, "text": text, "decoder": {"raw_text": raw_text},
                    "issues": [], "model_input": {"sha256": digest(prepared), "preprocessing": "raw"}}
        write_json(directory / (name + ".json"), {"kind": "issue_clip", "artifact_file": name, "metadata": metadata,
                   "payload_sha256": digest(clip), "file_sha256": digest((directory / name).read_bytes())})
        if with_input:
            model_name = request_id + ".f32le"
            (directory / model_name).write_bytes(prepared)
            write_json(directory / (model_name + ".json"), {"kind": "issue_blob", "artifact_file": model_name,
                       "metadata": metadata, "payload_sha256": digest(prepared)})
        return name

    def test_exact_source_placement_and_float_conversion_are_verified_without_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pcm = self.fixture(directory)
            self.add_request(directory, pcm, 1, 2)
            before = {p.name: p.read_bytes() for p in directory.iterdir()}
            result = audit_sessions.audit_session(directory)
            self.assertEqual(result["checks"]["raw_to_stream_compared"], 1)
            self.assertEqual(result["checks"]["raw_to_float_compared"], 1)
            self.assertFalse(any(v for k, v in result["checks"].items() if k.endswith("_mismatches")))
            self.assertEqual(result["errors"], [])
            self.assertEqual({p.name: p.read_bytes() for p in directory.iterdir()}, before)

    def test_corrupted_raw_and_float_artifacts_do_not_pass_placement_or_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pcm = self.fixture(directory)
            name = self.add_request(directory, pcm, 1, 2)
            write_wav(directory / name, b"\x00\x00" * 2)
            (directory / "u1-f.f32le").write_bytes(np.array([1, 2], dtype="<f4").tobytes())
            result = audit_sessions.audit_session(directory)
            for field in ("raw_payload_hash_mismatches", "request_hash_mismatches", "raw_to_stream_mismatches",
                          "model_input_hash_mismatches", "raw_to_float_mismatches"):
                self.assertEqual(result["checks"][field], 1, field)

    def test_filter_changes_partial_only_and_distinct_repeated_finals_are_review_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pcm = self.fixture(directory)
            self.add_request(directory, pcm, 1, 0, raw_text="שלום", text="")
            self.add_request(directory, pcm, 2, 2, raw_text="כן<|lt|>", text="כן<|lt|>")
            self.add_request(directory, pcm, 3, 4, raw_text="כן<|lt|>", text="כן<|lt|>")
            self.add_request(directory, pcm, 4, 6, partial=True, raw_text="מאוד מאוד", text="מאוד")
            result = audit_sessions.audit_session(directory)
            self.assertEqual(result["partial_without_saved_final_ids"], [4])
            self.assertEqual(result["filter_flag_counts"]["empty_after_filter"], 1)
            self.assertEqual(result["filter_flag_counts"]["changed_after_filter"], 1)
            self.assertEqual(result["filter_flag_counts"]["model_control_in_output"], 2)
            self.assertEqual(result["repeated_text_on_disjoint_finals"][0]["utterance_id"], 3)
            encoded = json.dumps(result, ensure_ascii=False)
            self.assertNotIn("שלום", encoded)
            self.assertNotIn("מאוד", encoded)

    def test_optional_decision_journal_reports_outcomes_without_copying_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runtime = directory / "runtime"
            runtime.mkdir()
            (runtime / "asr-decisions.jsonl").write_text(json.dumps({"decision": "accepted_final", "reason": "decoded",
                "utterance_id": 8, "request_id": "u8-f", "text": "PRIVATE", "raw_text": "PRIVATE"}) + "\n", encoding="utf-8")
            result = audit_sessions.audit_session(directory, runtime)
            self.assertEqual(result["decision_journals"][0]["decisions"], {"accepted_final": 1})
            self.assertNotIn("PRIVATE", json.dumps(result))

    def test_invalid_sidecar_and_escaped_artifact_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "bad.json").write_text("not json", encoding="utf-8")
            write_json(directory / "escape.json", {"kind": "stream_segment", "audio_file": "../outside.wav"})
            result = audit_sessions.audit_session(directory)
            self.assertEqual(len(result["errors"]), 2)
            self.assertIn("escapes", json.dumps(result["errors"]))

    def test_cli_only_writes_report_when_output_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "recordings/session"
            directory.mkdir(parents=True)
            pcm = self.fixture(directory)
            self.add_request(directory, pcm, 1, 0)
            args = ["--recordings-dir", str(root / "recordings"), "--runtime-dir", str(root / "runtime"), "--latest", "1"]
            before = sorted(p.name for p in root.iterdir())
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(audit_sessions.main(args), 0)
            self.assertEqual(sorted(p.name for p in root.iterdir()), before)
            self.assertNotIn("requests", json.loads(stdout.getvalue())["sessions"][0])
            report_path = root / "audit.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(audit_sessions.main([*args, "--output", str(report_path)]), 0)
            report = json.loads(report_path.read_text())
            self.assertEqual(len(report["sessions"][0]["requests"]), 1)
            self.assertFalse(report["model_inference_run"])
            self.assertFalse(report["network_accessed"])

    def test_latest_skips_newer_empty_sessions_unless_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recordings = root / "recordings"
            real = recordings / "real-audio"
            real.mkdir(parents=True)
            self.fixture(real)
            os.utime(real, (100, 100))
            for index in range(5):
                empty = recordings / f"new-empty-{index}"
                empty.mkdir()
                os.utime(empty, (200 + index, 200 + index))
            args = ["--recordings-dir", str(recordings), "--runtime-dir", str(root / "runtime"), "--latest", "1"]

            def run(extra):
                with redirect_stdout(io.StringIO()) as stdout:
                    self.assertEqual(audit_sessions.main([*args, *extra]), 0)
                return json.loads(stdout.getvalue())

            default = run([])
            self.assertEqual(default["sessions"][0]["session_id"], "real-audio")
            self.assertEqual(default["saved_audio_session_count"], 1)
            included = run(["--include-empty"])
            self.assertEqual(included["sessions"][0]["session_id"], "new-empty-4")
            self.assertEqual(included["saved_audio_session_count"], 0)
            self.assertIn("No saved-audio", included["selection_note"])
            explicit = run(["--session", "new-empty-0"])
            self.assertEqual(explicit["sessions"][0]["session_id"], "new-empty-0")


if __name__ == "__main__":
    unittest.main()
