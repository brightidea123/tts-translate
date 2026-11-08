"""Resolve optional model libraries only when a backend actually uses them.

The CLI, capture setup and worker spawning must not import every model runtime.
Missing packages keep the older ``None`` convention; installed packages retain
normal attribute/call behavior through a small thread-safe lazy reference.
"""

from __future__ import annotations

import importlib
import importlib.util
import threading
from typing import Any, Callable


class LazyDependency:
    def __init__(self, module_name: str, attribute: str | None = None,
                 configure: Callable[[Any], None] | None = None):
        self.module_name = module_name
        self.attribute = attribute
        self.configure = configure
        self._target: Any = None
        self._error: Exception | None = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._target is not None

    def load(self):
        with self._lock:
            if self._error is not None:
                raise self._error
            if self._target is None:
                try:
                    module = importlib.import_module(self.module_name)
                    if self.configure is not None:
                        self.configure(module)
                    self._target = getattr(module, self.attribute) if self.attribute else module
                except Exception as exc:
                    # Repeated failed native initialization can abort a process.
                    # Keep the first error rather than retrying the import.
                    self._error = exc
                    raise
            return self._target

    def __getattr__(self, name: str):
        return getattr(self.load(), name)

    def __call__(self, *args, **kwargs):
        return self.load()(*args, **kwargs)

    def __repr__(self) -> str:
        suffix = f".{self.attribute}" if self.attribute else ""
        return f"<lazy dependency {self.module_name}{suffix}; loaded={self.loaded}>"


def optional_dependency(module_name: str, attribute: str | None = None,
                        configure: Callable[[Any], None] | None = None):
    if importlib.util.find_spec(module_name) is None:
        return None
    return LazyDependency(module_name, attribute, configure)
