from __future__ import annotations

from copy import deepcopy
import io
import json
import os
import threading
import unittest

from tts_runtime.web_transport import HeadlessTransport, build_web_snapshot


class HeadlessTransportTests(unittest.TestCase):
    def setUp(self):
        self.reader, self.writer = os.pipe()
        self.output = io.StringIO()
        self.commands = []

        def handle(command):
            self.commands.append(command)
            return {"accepted": True, "message": "queued"}

        self.handler = handle
        self.transport = HeadlessTransport(self.reader, self.output, handle)

    def tearDown(self):
        os.close(self.reader)
        if self.writer is not None:
            os.close(self.writer)

    def messages(self):
        return [json.loads(line) for line in self.output.getvalue().splitlines()]

    def write(self, *commands):
        raw = "".join(json.dumps(command, ensure_ascii=False) + "\n" for command in commands)
        os.write(self.writer, raw.encode("utf-8"))

    def test_multiple_unicode_commands_drain_from_one_os_write(self):
        phrase = "שלום مرحبا <tag> (123)"
        self.write({"request_id": "one", "action": "reply", "text": phrase},
                   {"request_id": "two", "action": "flush"})
        self.assertFalse(self.transport.poll(0))
        self.assertEqual(self.commands[0]["text"], phrase)
        self.assertEqual([item["request_id"] for item in self.messages()], ["one", "two"])

    def test_split_multibyte_and_complete_lines_do_not_wait_for_more_input(self):
        raw = (json.dumps({"request_id": "split", "action": "reply", "text": "ש"}, ensure_ascii=False) + "\n").encode()
        split = raw.index("ש".encode()) + 1
        os.write(self.writer, raw[:split])
        self.assertFalse(self.transport.poll(0))
        self.assertEqual(self.messages(), [])
        os.write(self.writer, raw[split:])
        self.assertFalse(self.transport.poll(0))
        self.assertEqual(self.commands[0]["text"], "ש")

    def test_only_sixteen_commands_run_per_poll_and_buffer_drains_without_new_write(self):
        self.write(*({"request_id": str(index), "action": "flush"} for index in range(20)))
        self.assertFalse(self.transport.poll(0))
        self.assertEqual(len(self.commands), 16)
        self.assertFalse(self.transport.poll(0))
        self.assertEqual(len(self.commands), 20)

    def test_invalid_commands_acknowledge_failure_and_later_valid_command_survives(self):
        os.write(self.writer, b"not json\n[1,2]\n" + b'{"request_id":true,"action":"reply"}\n')
        self.write({"request_id": "bad-action", "action": "execute"},
                   {"request_id": "good", "action": "mark"})
        self.assertFalse(self.transport.poll(0))
        self.assertEqual([item["accepted"] for item in self.messages()], [False, False, False, False, True])
        self.assertEqual([command["action"] for command in self.commands], ["mark"])

    def test_oversized_and_invalid_utf8_lines_recover_at_the_next_newline(self):
        self.transport = HeadlessTransport(self.reader, self.output, self.handler, maxline=80)
        os.write(self.writer, b"x" * 100 + b"\n\xff\n")
        self.write({"request_id": "good", "action": "flush"})
        self.assertFalse(self.transport.poll(0))
        messages = self.messages()
        self.assertEqual([item["accepted"] for item in messages], [False, False, True])
        self.assertIn("80-byte", messages[0]["message"])
        self.assertIn("UTF-8", messages[1]["message"])

    def test_handler_failure_does_not_stop_transport_and_stop_ack_precedes_return(self):
        def handle(command):
            if command["action"] == "reply":
                raise RuntimeError("stage failed")
            return {"accepted": True, "stop": True}

        self.transport = HeadlessTransport(self.reader, self.output, handle)
        self.write({"request_id": "bad", "action": "reply"}, {"request_id": "stop", "action": "stop"})
        self.assertTrue(self.transport.poll(0))
        messages = self.messages()
        self.assertFalse(messages[0]["accepted"])
        self.assertIn("stage failed", messages[0]["message"])
        self.assertEqual(messages[-1], {"type": "ack", "request_id": "stop", "accepted": True, "message": ""})

    def test_eof_drains_final_unterminated_command(self):
        os.write(self.writer, b'{"request_id":"last","action":"mark"}')
        os.close(self.writer)
        self.writer = None
        self.assertFalse(self.transport.poll(0))
        self.assertTrue(self.transport.poll(0))
        self.assertEqual(self.messages()[0]["request_id"], "last")

    def test_eof_after_newline_does_not_invent_an_empty_command(self):
        self.write({"request_id": "last", "action": "flush"})
        os.close(self.writer)
        self.writer = None
        self.assertFalse(self.transport.poll(0))
        self.assertTrue(self.transport.poll(0))
        self.assertEqual(len(self.messages()), 1)

    def test_concurrent_state_publish_and_command_ack_are_atomic_json_lines(self):
        threads = [threading.Thread(target=lambda worker=worker: [self.transport.publish({"worker": worker, "value": value, "text": "עברית العربية"}) for value in range(80)]) for worker in range(4)]
        for thread in threads:
            thread.start()
        self.write({"request_id": "ack", "action": "flush"})
        self.transport.poll(0)
        for thread in threads:
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        messages = self.messages()
        self.assertEqual(len(messages), 321)
        states = [item["state"] for item in messages if item["type"] == "state"]
        self.assertEqual(len({(item["worker"], item["value"]) for item in states}), 320)
        self.assertTrue(all(item["text"] == "עברית العربية" for item in states))


class WebSnapshotTests(unittest.TestCase):
    def test_rtl_text_markup_and_punctuation_are_preserved_without_console_labels(self):
        source = "שלום (123), مرحباً <note> x < y\nשורה חדשה"
        translated = "مرحباً بالعالم؛ שלום!"
        ui = {
            "left_done": [{"entry_id": 7, "src": f"🗣️ HE: \x1b[32m{source}\x1b[0m", "dst": f"🌍 AR: \x1b[97m{translated}\x1b[0m"}],
            "right_active": {9: {"src": "⌨️ AR: كيف حالك؟", "dst": "🌍 HE~[1/2]: מה שלומך?", "src_final": True, "dst_final": False}},
            "right_order": [9], "status": "\x1b[31mready\x1b[0m", "stages": {"asr": "ready"},
        }
        original = deepcopy(ui)
        config = {"source_language": "HE", "target_language": "AR", "nested": {"same": True}}
        snapshot = build_web_snapshot(ui, config, "session", True, "issues", "/local/path")
        self.assertEqual(snapshot["transcript"][0]["source"], source)
        self.assertEqual(snapshot["transcript"][0]["translation"], translated)
        self.assertTrue(snapshot["transcript"][0]["source_final"])
        reply = snapshot["replies"][0]
        self.assertEqual((reply["source_language"], reply["target_language"]), ("AR", "HE"))
        self.assertEqual(reply["translation"], "מה שלומך?")
        self.assertFalse(reply["translation_final"])
        self.assertEqual(snapshot["status"], "ready")
        snapshot["config"]["nested"]["same"] = False
        self.assertTrue(config["nested"]["same"])
        self.assertEqual(ui, original)

    def test_history_is_bounded_and_active_order_is_kept_without_fake_entries(self):
        ui = {"left_done": [{"src": f"TR: phrase {index}", "dst": f"EN: value {index}"} for index in range(130)],
              "left_active": {index: {"src": f"TR~: active {index}"} for index in range(40)},
              "left_order": list(range(40))}
        snapshot = build_web_snapshot(ui, {"source_language": "TR", "target_language": "EN"}, None, False, "off", None)
        records = snapshot["transcript"]
        self.assertEqual(len(records), 120)
        self.assertEqual(records[0]["id"], "done-30")
        self.assertEqual(records[-20]["source"], "active 20")
        self.assertEqual(records[-1]["source"], "active 39")
        self.assertEqual(snapshot["replies"], [])
        self.assertEqual(snapshot["stages"], {})
        self.assertIsNone(snapshot["recording_directory"])

    def test_refined_active_id_replaces_matching_history_without_reordering_text(self):
        ui = {"left_done": [{"entry_id": 2, "src": "old", "dst": "old translation"}],
              "left_active": {"2": {"src": "HE✓: חדש", "dst": "AR~: جديد"}}, "left_order": [2]}
        records = build_web_snapshot(ui, {"source_language": "HE", "target_language": "AR"}, "s", True, "full", "recordings")["transcript"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], "2")
        self.assertEqual(records[0]["source"], "חדש")


if __name__ == "__main__":
    unittest.main()
