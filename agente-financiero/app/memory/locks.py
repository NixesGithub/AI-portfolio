"""Cerrojo por conversación.

Dos mensajes simultáneos al mismo ``conversation_id`` intercalarían turnos y
dejarían el historial incoherente (por ejemplo, un ``ToolMessage`` sin su
``AIMessage``). Serializamos por id, no globalmente: conversaciones distintas
siguen corriendo en paralelo.

Es un cerrojo de proceso. Con varias réplicas haría falta uno distribuido
(Redis, Postgres advisory locks) detrás de esta misma interfaz.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncIterator


class ConversationLocks:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._waiters: dict[str, int] = defaultdict(int)
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, conversation_id: str) -> AsyncIterator[None]:
        async with self._guard:
            lock = self._locks.setdefault(conversation_id, asyncio.Lock())
            self._waiters[conversation_id] += 1
        try:
            async with lock:
                yield
        finally:
            # Sin recuento de espera el diccionario crecería sin límite.
            async with self._guard:
                self._waiters[conversation_id] -= 1
                if self._waiters[conversation_id] <= 0:
                    self._waiters.pop(conversation_id, None)
                    self._locks.pop(conversation_id, None)


class NullLocks:
    """Sin serialización: útil en tests que no comparten conversación."""

    @asynccontextmanager
    async def acquire(self, conversation_id: str) -> AsyncIterator[None]:
        yield
