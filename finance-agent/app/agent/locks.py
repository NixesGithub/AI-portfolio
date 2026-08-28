"""Locks por clave, con limpieza.

Dos POST simultáneos al mismo `conversation_id` harían cada uno "leer
historial → llamar al modelo → escribir historial" en paralelo, y el segundo
guardaría un turno construido sobre un historial que ya quedó viejo. Serializar
por conversación evita ese intercalado sin serializar el servicio entero:
conversaciones distintas siguen corriendo en paralelo.

Un `dict[str, Lock]` a secas alcanzaría, pero crece para siempre. Este lleva
la cuenta de quién lo está esperando y borra la entrada cuando queda libre.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


class KeyedLocks:
    def __init__(self) -> None:
        self._locks: dict[str, tuple[asyncio.Lock, int]] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, key: str):
        async with self._guard:
            lock, waiters = self._locks.get(key, (asyncio.Lock(), 0))
            self._locks[key] = (lock, waiters + 1)

        try:
            async with lock:
                yield
        finally:
            async with self._guard:
                current_lock, waiters = self._locks[key]
                if waiters <= 1:
                    del self._locks[key]
                else:
                    self._locks[key] = (current_lock, waiters - 1)

    def __len__(self) -> int:
        return len(self._locks)
