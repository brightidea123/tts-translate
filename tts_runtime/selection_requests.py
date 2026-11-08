"""Bounded, session-scoped selection lookups; no model or audio imports."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import threading
import time
from uuid import uuid4


LANGUAGES = frozenset({"AR", "HE", "DE", "TR", "FR", "RU", "EN", "ES"})


class SelectionRequests:
    """Keep tooltip state outside live snapshots and reuse identical lookups.

    One core command serves concurrent requests for the same text/direction.
    Every caller still receives its own selection ID. Resetting the generation
    drops all records/cache so late packets cannot cross a session boundary.
    """

    def __init__(self, *, max_pending=4, max_records=128, max_cache=128,
                 timeout=120.0, clock=time.monotonic):
        if min(max_pending, max_records, max_cache) < 1 or max_pending > max_records or timeout <= 0:
            raise ValueError("Selection request limits must be positive and bounded")
        self.max_pending, self.max_records, self.max_cache = max_pending, max_records, max_cache
        self.timeout, self.clock = timeout, clock
        self.generation = None
        self.records, self.cache = OrderedDict(), OrderedDict()
        self._lock = threading.RLock()

    def reset(self, generation=None):
        with self._lock:
            self.generation = generation
            self.records.clear()
            self.cache.clear()

    @staticmethod
    def validate(config):
        if not isinstance(config, dict):
            raise ValueError("Selection request must be an object")
        text = config.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 1500 or "\x00" in text:
            raise ValueError("Selected text must contain 1–1500 characters without NUL")
        selection = config.get("selection_id")
        if (not isinstance(selection, str) or not 1 <= len(selection) <= 128
                or any(ord(char) < 32 or ord(char) == 127 for char in selection)):
            raise ValueError("selection_id must contain 1–128 characters without control characters")
        languages = []
        for name in ("source_language", "target_language"):
            value = config.get(name)
            if not isinstance(value, str) or value.strip().upper() not in LANGUAGES:
                raise ValueError("Selection language must be AR, HE, DE, TR, FR, RU, EN, or ES")
            languages.append(value.strip().upper())
        if languages[0] == languages[1]:
            raise ValueError("Selection languages must differ")
        return {"text": text.strip(), "selection_id": selection,
                "source_language": languages[0], "target_language": languages[1]}

    @staticmethod
    def _public(record):
        return deepcopy({key: value for key, value in record.items()
                         if key not in {"key", "created_at", "primary_id", "text"}})

    def _expire(self):
        now = self.clock()
        expired = {record["primary_id"] for record in self.records.values()
                   if record["status"] == "pending" and now - record["created_at"] >= self.timeout}
        for record in self.records.values():
            if record["status"] == "pending" and record["primary_id"] in expired:
                record.update(status="error", error="Selection translation timed out; select the phrase again")

    def reserve(self, config):
        config = self.validate(config)
        with self._lock:
            self._expire()
            if self.generation is None:
                raise RuntimeError("Start a session before translating a selection")
            key = (self.generation, config["source_language"], config["target_language"], config["text"])
            cached = self.cache.get(key)
            if cached is None and sum(row["status"] == "pending" for row in self.records.values()) >= self.max_pending:
                raise RuntimeError("Selection translation is busy; wait for a pending phrase")
            primary = next((row for row in self.records.values()
                            if row["key"] == key and row["status"] == "pending"), None)
            lookup_id = uuid4().hex
            record = {**config, "lookup_id": lookup_id, "status": "complete" if cached is not None else "pending",
                      "translation": cached or "", "error": "", "key": key,
                      "created_at": self.clock(), "primary_id": primary["primary_id"] if primary else lookup_id}
            if cached is not None:
                self.cache.move_to_end(key)
            self.records[lookup_id] = record
            while len(self.records) > self.max_records:
                oldest = next((identity for identity, row in self.records.items() if row["status"] != "pending"), None)
                if oldest is None:
                    break
                self.records.pop(oldest)
            return self._public(record), cached is None and primary is None, config

    def result(self, lookup_id):
        with self._lock:
            self._expire()
            record = self.records.get(lookup_id)
            return self._public(record) if record is not None else None

    def fail(self, lookup_id, message):
        with self._lock:
            self._expire()
            record = self.records.get(lookup_id)
            if record is None:
                return
            for row in self.records.values():
                if row["primary_id"] == record["primary_id"] and row["status"] == "pending":
                    row.update(status="error", error=str(message)[:1000])

    def complete(self, generation, packet):
        with self._lock:
            self._expire()
            record = self.records.get(packet.get("lookup_id"))
            if (generation != self.generation or record is None or record["status"] != "pending"
                    or record["lookup_id"] != record["primary_id"]):
                return False
            if any(packet.get(name) != record[name] for name in ("selection_id", "source_language", "target_language")):
                return False
            error = packet.get("error")
            text = packet.get("translation")
            if error:
                self.fail(record["lookup_id"], error)
                return True
            if not isinstance(text, str) or not text.strip() or len(text) > 12000:
                self.fail(record["lookup_id"], "The translation worker returned an empty or invalid selection result")
                return True
            self.cache[record["key"]] = text.strip()
            self.cache.move_to_end(record["key"])
            while len(self.cache) > self.max_cache:
                self.cache.popitem(last=False)
            for row in self.records.values():
                if row["primary_id"] == record["lookup_id"] and row["status"] == "pending":
                    row.update(status="complete", translation=text.strip(), error="")
            return True
