"""Regression for pasted Unicode/Enter being stranded in TextIO read-ahead."""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch


class TerminalInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "tts-transcribe.py"
        with patch.object(sys, "argv", [str(path)]):
            cls.module = runpy.run_path(str(path), run_name="terminal_input_test")

    def setUp(self):
        self.reader, self.writer = os.pipe()
        self.globals = self.module["read_key"].__globals__
        self.old_fd = self.globals["stdin_fd"]
        self.globals["stdin_fd"] = self.reader
        self.globals["stdin_characters"].clear()
        self.globals["stdin_decoder"].reset()

    def tearDown(self):
        self.globals["stdin_fd"] = self.old_fd
        os.close(self.reader)
        if self.writer is not None:
            os.close(self.writer)

    def test_entire_paste_and_enter_drain_without_another_os_write(self):
        phrase = "שלום مرحبا Привет test\r"
        os.write(self.writer, phrase.encode("utf-8"))
        read_key = self.module["read_key"]
        self.assertEqual("".join(read_key(0) for _ in phrase), phrase)
        self.assertIsNone(read_key(0))

    def test_a_multibyte_character_split_across_reads_decodes_once(self):
        encoded = "ש".encode("utf-8")
        os.write(self.writer, encoded[:1])
        self.assertIsNone(self.module["read_key"](0))
        os.write(self.writer, encoded[1:] + b"\r")
        self.assertEqual(self.module["read_key"](0), "ש")
        self.assertEqual(self.module["read_key"](0), "\r")
        self.assertIsNone(self.module["read_key"](0))

    def test_terminal_eof_requests_exit_instead_of_spinning(self):
        os.close(self.writer)
        self.writer = None
        self.assertEqual(self.module["read_key"](0), "\x1b")


if __name__ == "__main__":
    unittest.main()
