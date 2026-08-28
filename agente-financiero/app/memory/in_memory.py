"""Almacén en proceso. Por defecto en local y en los tests.

Es rápido y no necesita nada, pero muere con el proceso y no se comparte entre
réplicas: en producción se usa el backend ``sqlite`` (o el Redis/Postgres que
toque, implementando la misma interfaz).
"""

from __future__ import annotations

import asyncio
from typing import Sequence

from langchain_core.messages import BaseMessage

from app.memory.base import (
    ConversationMeta,
    ConversationStore,
    StoredMessage,
    new_id,
    utcnow,
)


class InMemoryConversationStore(ConversationStore):
    def __init__(self, max_messages: int = 0) -> None:
        self._max_messages = max_messages
        self._threads: dict[str, list[StoredMessage]] = {}
        self._meta: dict[str, ConversationMeta] = {}
        self._lock = asyncio.Lock()

    async def append(
        self, conversation_id: str, messages: Sequence[BaseMessage]
    ) -> list[StoredMessage]:
        now = utcnow()
        stored = [
            StoredMessage(
                id=new_id("msg"),
                conversation_id=conversation_id,
                created_at=now,
                message=message,
            )
            for message in messages
        ]
        async with self._lock:
            thread = self._threads.setdefault(conversation_id, [])
            thread.extend(stored)
            if self._max_messages and len(thread) > self._max_messages:
                del thread[: len(thread) - self._max_messages]
            previous = self._meta.get(conversation_id)
            self._meta[conversation_id] = ConversationMeta(
                conversation_id=conversation_id,
                created_at=previous.created_at if previous else now,
                updated_at=now,
                message_count=len(thread),
            )
        return stored

    async def get(
        self, conversation_id: str, *, limit: int | None = None
    ) -> list[StoredMessage]:
        async with self._lock:
            thread = list(self._threads.get(conversation_id, []))
        return thread[-limit:] if limit else thread

    async def meta(self, conversation_id: str) -> ConversationMeta | None:
        async with self._lock:
            return self._meta.get(conversation_id)

    async def delete(self, conversation_id: str) -> bool:
        async with self._lock:
            existed = self._threads.pop(conversation_id, None) is not None
            self._meta.pop(conversation_id, None)
        return existed
