"""ConversationStore en memoria.

Sirve para los tests y para levantar el servicio sin disco. No es una opción
de producción: al reiniciar el proceso se pierde todo.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from langchain_core.messages import BaseMessage

from .base import Conversation, ConversationNotFound, MessageRecord


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._threads: dict[str, list[MessageRecord]] = {}
        self._created: dict[str, datetime] = {}
        self._updated: dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def load(self, conversation_id: str) -> Conversation:
        if conversation_id not in self._threads:
            raise ConversationNotFound(conversation_id)
        return Conversation(
            id=conversation_id,
            created_at=self._created[conversation_id],
            updated_at=self._updated[conversation_id],
            messages=list(self._threads[conversation_id]),
        )

    async def history(self, conversation_id: str) -> list[BaseMessage]:
        return [r.message for r in self._threads.get(conversation_id, [])]

    async def append(
        self, conversation_id: str, messages: list[BaseMessage] | tuple[BaseMessage, ...]
    ) -> None:
        if not messages:
            return
        now = datetime.now(timezone.utc)
        async with self._lock:
            thread = self._threads.setdefault(conversation_id, [])
            self._created.setdefault(conversation_id, now)
            self._updated[conversation_id] = now
            start = thread[-1].seq + 1 if thread else 0
            thread.extend(
                MessageRecord(seq=start + offset, message=message, created_at=now)
                for offset, message in enumerate(messages)
            )

    async def close(self) -> None:  # simetría con el store de SQLite
        return None
