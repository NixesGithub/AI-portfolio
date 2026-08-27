"""Implementación de ConversationStore sobre SQLite.

Por qué SQLite y no Postgres o Redis: el enunciado pide una app que corra en
Docker y mantenga historiales por id. SQLite con un volumen montado da
durabilidad real (sobrevive a un reinicio del contenedor) sin sumar un
servicio más al compose. El límite conocido es un solo proceso escritor; está
documentado en el README junto con la ruta de migración.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone

import aiosqlite
from langchain_core.messages import BaseMessage

from .base import Conversation, ConversationNotFound, MessageRecord
from .serde import dumps, loads

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    conversation_id TEXT    NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,
    type            TEXT    NOT NULL,
    payload         TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    PRIMARY KEY (conversation_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages (conversation_id, seq);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SQLiteConversationStore:
    """Store persistente. Instanciar con `await SQLiteConversationStore.create(path)`."""

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._db = connection
        # SQLite serializa escrituras igual, pero un lock propio evita que dos
        # `append` concurrentes se pisen entre el cálculo del `seq` y el INSERT.
        self._write_lock = asyncio.Lock()

    @classmethod
    async def create(cls, path: str) -> "SQLiteConversationStore":
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)

        connection = await aiosqlite.connect(path)
        # WAL: los lectores no bloquean al escritor. Con un endpoint de lectura
        # (GET /chat/{id}) y otro de escritura sobre la misma base, importa.
        await connection.execute("PRAGMA journal_mode=WAL")
        await connection.execute("PRAGMA foreign_keys=ON")
        await connection.execute("PRAGMA busy_timeout=5000")
        await connection.executescript(_SCHEMA)
        await connection.commit()
        log.info("sqlite store listo", extra={"path": path})
        return cls(connection)

    async def close(self) -> None:
        await self._db.close()

    # -- lectura -------------------------------------------------------------

    async def load(self, conversation_id: str) -> Conversation:
        async with self._db.execute(
            "SELECT created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise ConversationNotFound(conversation_id)

        records = await self._records(conversation_id)
        return Conversation(
            id=conversation_id,
            created_at=datetime.fromisoformat(row[0]),
            updated_at=datetime.fromisoformat(row[1]),
            messages=records,
        )

    async def history(self, conversation_id: str) -> list[BaseMessage]:
        return [record.message for record in await self._records(conversation_id)]

    async def _records(self, conversation_id: str) -> list[MessageRecord]:
        async with self._db.execute(
            "SELECT seq, payload, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY seq",
            (conversation_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            MessageRecord(
                seq=row[0],
                message=loads(row[1]),
                created_at=datetime.fromisoformat(row[2]),
            )
            for row in rows
        ]

    # -- escritura -----------------------------------------------------------

    async def append(
        self, conversation_id: str, messages: list[BaseMessage] | tuple[BaseMessage, ...]
    ) -> None:
        if not messages:
            return

        now = _now().isoformat()
        async with self._write_lock:
            await self._db.execute("BEGIN IMMEDIATE")
            try:
                await self._db.execute(
                    "INSERT INTO conversations (id, created_at, updated_at) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at",
                    (conversation_id, now, now),
                )
                async with self._db.execute(
                    "SELECT COALESCE(MAX(seq), -1) FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ) as cursor:
                    (last_seq,) = await cursor.fetchone()

                await self._db.executemany(
                    "INSERT INTO messages (conversation_id, seq, type, payload, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (conversation_id, last_seq + offset, message.type,
                         dumps(message), now)
                        for offset, message in enumerate(messages, start=1)
                    ],
                )
                await self._db.commit()
            except Exception:
                await self._db.rollback()
                raise
