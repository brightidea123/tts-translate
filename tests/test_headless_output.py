import io
import json
import os
import threading
import time
import unittest

from tts_runtime.web_transport import HeadlessTransport


class BlockedStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.entered, self.release = threading.Event(), threading.Event()

    def write(self, value):
        self.entered.set()
        self.release.wait(3)
        return super().write(value)


class AsyncOutputTests(unittest.TestCase):
    def setUp(self):
        self.reader, self.writer = os.pipe()
        self.addCleanup(os.close, self.reader)
        self.addCleanup(os.close, self.writer)

    def transport(self, stream, **kwargs):
        value = HeadlessTransport(self.reader, stream, lambda command: {"accepted": True}, async_output=True, **kwargs)
        self.addCleanup(value.close)
        return value

    def test_slow_output_does_not_block_capture_state_or_input_and_ack_precedes_latest_state(self):
        stream = BlockedStream()
        transport = self.transport(stream, state_interval=0.01)
        try:
            transport.publish({"counter": 0})
            self.assertTrue(stream.entered.wait(1))
            started = time.monotonic()
            for counter in range(1, 101):
                transport.publish({"counter": counter, "text": "שלום مرحبا"})
            os.write(self.writer, b'{"request_id":"one","action":"lookup"}\n')
            self.assertFalse(transport.poll(0))
            self.assertLess(time.monotonic() - started, 0.2)
            self.assertGreater(transport.snapshot["state_coalesced"], 90)
        finally:
            stream.release.set()
            transport.close(timeout=1)
        messages = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual([message["type"] for message in messages], ["state", "ack", "state"])
        self.assertEqual(messages[-1]["state"]["counter"], 100)
        self.assertEqual(messages[-1]["state"]["text"], "שלום مرحبا")
        self.assertFalse(transport.snapshot["writer_alive"])

    def test_critical_capacity_failure_is_explicit_bounded_and_stops_input(self):
        stream = BlockedStream()
        transport = self.transport(stream, critical_capacity=2)
        try:
            transport.publish({"counter": 0}); self.assertTrue(stream.entered.wait(1))
            self.assertTrue(transport.send_packet({"type": "ack", "id": 1}))
            self.assertTrue(transport.send_packet({"type": "ack", "id": 2}))
            self.assertFalse(transport.send_packet({"type": "ack", "id": 3}))
            self.assertEqual(transport.snapshot["critical_pending"], 2)
            self.assertTrue(transport.poll(0))
            started = time.monotonic(); transport.close(timeout=0.01)
            self.assertLess(time.monotonic() - started, 0.1)
        finally:
            stream.release.set(); transport.close(timeout=1)

    def test_packets_are_snapshots_and_critical_unicode_events_are_atomic(self):
        stream = BlockedStream()
        transport = self.transport(stream)
        try:
            transport.publish({"first": True}); self.assertTrue(stream.entered.wait(1))
            packet = {"type": "lookup_result", "translation": "שלום", "details": {"revision": 1}}
            self.assertTrue(transport.send_packet(packet))
            packet["details"]["revision"] = 2
        finally:
            stream.release.set(); transport.close(timeout=1)
        lines = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(lines[-1]["translation"], "שלום")
        self.assertEqual(lines[-1]["details"]["revision"], 1)
