"""Read actual recording artifacts; no models, microphones, or playback."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
import wave
from array import array

from tts_runtime.audio_evidence import AudioEvidence, create_session_directory


def pcm(*samples):
    return struct.pack("<" + "h" * len(samples), *samples)


def wave_samples(path):
    with wave.open(str(path), "rb") as audio:
        assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
        payload = audio.readframes(audio.getnframes())
    return list(struct.unpack("<" + "h" * (len(payload) // 2), payload))


def events(writer):
    return [json.loads(line) for line in writer.events_path.read_text(encoding="utf-8").splitlines()]


class HeldWriter(AudioEvidence):
    def __init__(self, *args, **kwargs):
        self.entered, self.release = threading.Event(), threading.Event()
        super().__init__(*args, **kwargs)

    def _write_capture(self, item, handle):
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError("fixture did not release writer")
        super()._write_capture(item, handle)


class AudioEvidenceTests(unittest.TestCase):
    def test_open_stream_part_wav_is_readable_before_orderly_shutdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = AudioEvidence(temporary, full_stream=True, sample_rate=8)
            writer.capture(pcm(1, 2, 3))
            writer.capture(pcm(4, 5))
            self.assertTrue(writer.flush())
            part = next(Path(temporary).glob("*.wav.part"))
            self.assertEqual(wave_samples(part), [1, 2, 3, 4, 5])
            starts = [e for e in events(writer) if e["kind"] == "stream_segment_started"]
            self.assertEqual(starts[0]["start_sample"], 0)
            self.assertTrue(writer.close())

    def test_rotated_capture_preserves_every_real_sample_and_source_offset(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = AudioEvidence(temporary, full_stream=True, sample_rate=8, segment_seconds=0.5)
            self.assertTrue(writer.capture(pcm(0, 1, 2, 3, 4, 5), start_sample=100, timestamp_ns=9000))
            self.assertTrue(writer.capture(pcm(6, 7, 8, 9), start_sample=106, timestamp_ns=750009000))
            self.assertTrue(writer.close())
            paths = sorted(Path(temporary).glob("*.wav"))
            self.assertEqual([wave_samples(path) for path in paths], [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])
            records = [json.loads(path.with_suffix(".json").read_text()) for path in paths]
            self.assertEqual([(r["start_sample"], r["end_sample"]) for r in records], [(100, 104), (104, 108), (108, 110)])
            for path, record in zip(paths, records):
                self.assertEqual(record["file_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(writer.snapshot["written_capture_samples"], 10)

    def test_source_gap_and_overlap_never_silently_join_distinct_intervals(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = AudioEvidence(temporary, full_stream=True, sample_rate=8)
            writer.capture(pcm(1, 2, 3, 4), start_sample=0)
            writer.capture(pcm(9, 10), start_sample=10)
            writer.capture(pcm(11, 12), start_sample=11)
            self.assertTrue(writer.close())
            self.assertEqual([wave_samples(p) for p in sorted(Path(temporary).glob("*.wav"))], [[1, 2, 3, 4], [9, 10], [11, 12]])
            discontinuities = [e for e in events(writer) if e["kind"] == "stream_discontinuity"]
            self.assertEqual([e["delta_samples"] for e in discontinuities], [6, -1])
            self.assertEqual([e["reason"] for e in discontinuities], ["gap", "overlap_or_reordered_input"])

    def test_capture_copies_mutable_input_and_returns_while_writer_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = HeldWriter(temporary, full_stream=True, sample_rate=8, queue_seconds=0.5)
            source = bytearray(pcm(1, 2, 3, 4))
            self.assertTrue(writer.capture(source))
            self.assertTrue(writer.entered.wait(2))
            source[:] = pcm(99, 99, 99, 99)
            self.assertTrue(writer.capture(pcm(5, 6, 7, 8)))
            returned = threading.Event()
            accepted = []

            def overflow():
                accepted.append(writer.capture(pcm(9, 10, 11, 12)))
                returned.set()

            caller = threading.Thread(target=overflow)
            caller.start()
            self.assertTrue(returned.wait(1), "capture waited for the blocked disk writer")
            caller.join(1)
            self.assertEqual(accepted, [False])
            writer.release.set()
            self.assertTrue(writer.close())
            self.assertEqual(wave_samples(next(Path(temporary).glob("*.wav"))), list(range(1, 9)))
            drops = [e for e in events(writer) if e["kind"] == "recording_drop"]
            self.assertEqual([(e["start_sample"], e["end_sample"]) for e in drops], [(8, 12)])
            self.assertEqual(writer.snapshot["rejected_capture_samples"], 4)

    def test_flagged_clips_and_float_blobs_are_exact_and_flush_publishes_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = AudioEvidence(temporary, role="asr", full_stream=False)
            raw = bytearray(pcm(1000, -2000, 32767))
            meta = {"utterance_id": 7, "language": "he", "text": "שלום", "reason_metric": {"avg_logprob": -1.2}}
            self.assertFalse(writer.save_clip(raw, meta), "normal operation must not retain unflagged audio")
            self.assertTrue(writer.save_clip(raw, meta, ["low_confidence"], start_sample=16, label="raw"))
            raw[:] = pcm(0, 0, 0)
            meta["text"] = "modified caller state"
            exact_float = struct.pack("<fff", 2.0, -4.25, 0.125)
            self.assertTrue(writer.save_blob(exact_float, {"dtype": "<f4"}, ["debug"], label="model-input"))
            self.assertTrue(writer.flush())
            wav_path = next(Path(temporary).glob("*.wav"))
            self.assertEqual(wave_samples(wav_path), [1000, -2000, 32767])
            clip = json.loads(Path(str(wav_path) + ".json").read_text())
            self.assertEqual(clip["metadata"]["text"], "שלום")
            self.assertEqual((clip["start_sample"], clip["end_sample"]), (16, 19))
            blob = next(Path(temporary).glob("*.f32le"))
            self.assertEqual(blob.read_bytes(), exact_float)
            self.assertEqual(json.loads(Path(str(blob) + ".json").read_text())["payload_sha256"], hashlib.sha256(exact_float).hexdigest())
            self.assertTrue(writer.close())

    def test_issue_budget_does_not_delete_existing_artifacts_or_stop_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            unrelated = Path(temporary) / "keep-me.txt"
            unrelated.write_text("user data")
            writer = AudioEvidence(temporary, full_stream=True, sample_rate=8, max_clip_bytes=4)
            self.assertTrue(writer.save_clip(pcm(1, 2), reasons=["clipped"]))
            self.assertFalse(writer.save_clip(pcm(3), reasons=["empty_result"]))
            self.assertTrue(writer.capture(pcm(5, 6, 7)))
            self.assertTrue(writer.close())
            self.assertEqual(writer.snapshot["written_artifacts"], 1)
            self.assertEqual(writer.snapshot["written_capture_samples"], 3)
            self.assertEqual(unrelated.read_text(), "user data")
            self.assertTrue(any(e["kind"] == "artifact_not_saved" for e in events(writer)))

    def test_unique_sessions_and_writers_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = create_session_directory(temporary)
            second = create_session_directory(temporary)
            self.assertNotEqual(first, second)
            one, two = AudioEvidence(first, role="asr"), AudioEvidence(first, role="asr")
            one.save_clip(pcm(1), reasons=["test"])
            two.save_clip(pcm(2), reasons=["test"])
            self.assertTrue(one.close())
            self.assertTrue(two.close())
            self.assertNotEqual(one.writer_id, two.writer_id)
            self.assertEqual(sorted(wave_samples(p)[0] for p in first.glob("*.wav")), [1, 2])
            self.assertEqual(len(list(first.glob("*-summary.json"))), 2)

    def test_flush_and_close_are_bounded_when_io_is_stalled(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = HeldWriter(temporary, full_stream=True, sample_rate=8)
            writer.capture(pcm(1, 2))
            self.assertTrue(writer.entered.wait(2))
            self.assertFalse(writer.flush(timeout=0))
            self.assertFalse(writer.close(timeout=0))
            writer.release.set()
            self.assertTrue(writer.close(timeout=2))
            self.assertTrue(writer.snapshot["closed"])

    def test_write_failure_is_not_reported_as_a_successful_flush(self):
        class FailingWriter(AudioEvidence):
            def _write_artifact(self, kind, item, events):
                raise OSError("fixture disk unavailable")

        with tempfile.TemporaryDirectory() as temporary:
            writer = FailingWriter(temporary)
            self.assertTrue(writer.save_clip(pcm(1, 2), reasons=["error"]))
            self.assertFalse(writer.flush(timeout=2))
            self.assertFalse(writer.close(timeout=2))
            self.assertIn("fixture disk unavailable", writer.snapshot["error"])
            summary = json.loads(writer.summary_path.read_text())
            self.assertIn("fixture disk unavailable", summary["error"])

    def test_hardware_discontinuity_splits_even_contiguous_received_offsets(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = AudioEvidence(temporary, full_stream=True)
            self.assertTrue(writer.capture(pcm(1, 2), start_sample=0, timestamp_ns=1000))
            self.assertTrue(writer.capture(pcm(3, 4), start_sample=2, timestamp_ns=9000000000, discontinuity=True))
            self.assertTrue(writer.close())
            files = sorted(Path(temporary).glob("*.wav"))
            self.assertEqual(len(files), 2)
            payloads = []
            for path in files:
                with wave.open(str(path), "rb") as audio:
                    payloads.append(audio.readframes(audio.getnframes()))
            self.assertEqual(payloads, [pcm(1, 2), pcm(3, 4)])
            records = [json.loads(line) for line in writer.events_path.read_text().splitlines()]
            gaps = [record for record in records if record.get("reason") == "input_device_or_clock_discontinuity"]
            self.assertEqual(len(gaps), 1)
            self.assertTrue(gaps[0]["missing_input_samples_unknown"])

    def test_rejects_incomplete_samples_and_path_traversal_suffixes(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = AudioEvidence(temporary, full_stream=True)
            with self.assertRaises(ValueError):
                writer.capture(b"\x01")
            with self.assertRaises(ValueError):
                writer.capture(array("H", [1, 2]))
            with self.assertRaises(ValueError):
                writer.capture(memoryview(pcm(1, 2, 3, 4)).cast("h", (2, 2)))
            with self.assertRaises(ValueError):
                writer.save_blob(b"raw", reasons=["test"], suffix="../../outside")
            self.assertTrue(writer.close())


if __name__ == "__main__":
    unittest.main()
