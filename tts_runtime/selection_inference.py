"""Picklable selection jobs, separate from conversation/translation/TTS state."""
from dataclasses import dataclass
from .selection_requests import SelectionRequests


@dataclass(frozen=True)
class SelectionTask:
    lookup_id: str
    selection_id: str
    session_id: str
    text: str
    source_language: str
    target_language: str

    @classmethod
    def from_command(cls, command, session_id):
        values = SelectionRequests.validate(command)
        identity = command.get("lookup_id")
        if not isinstance(identity, str) or len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("Invalid selection lookup ID")
        return cls(identity, values["selection_id"], session_id, values["text"],
                   values["source_language"], values["target_language"])


@dataclass(frozen=True)
class SelectionResult:
    task: SelectionTask
    translation: str = ""
    error: str = ""
