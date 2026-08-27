"""Caché TTL en memoria para respuestas de Yahoo.

Dentro de una conversación es normal que el modelo pida dos veces la misma
cotización (primero para responder, después para comparar). Sin caché eso son
dos llamadas de red a un proveedor que no controlamos y que rate-limitea.

Es deliberadamente un dict por proceso: con varias réplicas cada una tendrá su
copia, lo cual está bien para datos que caducan en segundos. Si algún día
molesta, se cambia por Redis detrás de la misma interfaz.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Hashable


class TTLCache:
    def __init__(self, ttl_seconds: float, max_entries: int = 512) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._entries: dict[Hashable, tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

    async def get_or_set(
        self, key: Hashable, factory: Callable[[], Awaitable[Any]]
    ) -> Any:
        now = time.monotonic()
        async with self._lock:
            hit = self._entries.get(key)
            if hit is not None and hit[0] > now:
                return hit[1]

        # El factory corre fuera del lock: si tarda 3 s en responder, no
        # queremos bloquear a todo el que consulte otro símbolo mientras tanto.
        # El costo es que dos pedidos simultáneos del mismo símbolo pueden
        # llamar a Yahoo dos veces; es más barato que serializar todo.
        value = await factory()

        async with self._lock:
            if len(self._entries) >= self._max_entries:
                self._evict_expired(time.monotonic())
            self._entries[key] = (time.monotonic() + self._ttl, value)
        return value

    def _evict_expired(self, now: float) -> None:
        expired = [k for k, (deadline, _) in self._entries.items() if deadline <= now]
        for key in expired:
            del self._entries[key]
        if len(self._entries) >= self._max_entries:
            # Todo vigente y lleno: tiramos la entrada más vieja.
            oldest = min(self._entries, key=lambda k: self._entries[k][0])
            del self._entries[oldest]

    def clear(self) -> None:
        self._entries.clear()
