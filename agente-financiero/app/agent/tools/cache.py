"""Caché TTL en memoria para las respuestas de Yahoo Finance.

Sin caché, tres preguntas seguidas sobre AAPL son tres llamadas a Yahoo. Con
ella, la cotización se reutiliza un minuto: menos latencia, menos riesgo de que
nos limiten por volumen y respuestas coherentes dentro de un mismo turno.
"""

from __future__ import annotations

import threading
import time
from typing import Any


class TTLCache:
    def __init__(self, max_entries: int = 512) -> None:
        self._max_entries = max_entries
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at < now:
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl_s: float) -> None:
        if ttl_s <= 0:
            return
        with self._lock:
            if len(self._data) >= self._max_entries:
                # Purga simple: fuera lo caducado y, si no basta, lo más antiguo.
                now = time.monotonic()
                for k, (expires_at, _) in list(self._data.items()):
                    if expires_at < now:
                        self._data.pop(k, None)
                while len(self._data) >= self._max_entries:
                    self._data.pop(next(iter(self._data)))
            self._data[key] = (time.monotonic() + ttl_s, value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
