"""Almacén en SQLite: sobrevive a los reinicios del contenedor.

SQLite no es la respuesta para un servicio con muchas réplicas, pero sí para un
único contenedor con un volumen: da durabilidad real sin añadir infraestructura,
y aísla el resto del código de la decisión (mañana puede ser Postgres). Las
llamadas son bloqueantes, así que van a un hilo para no parar el event loop.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Sequence

from langchain_core.messages import BaseMessage

from app.memory.base import (
    ConversationMeta,
    ConversationStore,
    StoredMessage,
    new_id,
    utcnow,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    payload         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages (conversation_id, seq);
"""


class SQLiteConversationStore(ConversationStore):
    def __init__(self, path: str, max_messages: int = 0) -> None:
        self._path = path
        self._max_messages = max_messages
        # Un único hilo escritor: SQLite serializa las escrituras de todas formas
        # y así no hay que pelearse con "database is locked".
        self._lock = asyncio.Lock()
        self._init_schema()

    # -- infraestructura ----------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # -- API ----------------------------------------------------------------
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
            await asyncio.to_thread(self._append_sync, conversation_id, stored, now)
        return stored

    def _append_sync(
        self, conversation_id: str, stored: list[StoredMessage], now: datetime
    ) -> None:
        iso = now.isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (conversation_id, created_at, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(conversation_id) DO UPDATE SET updated_at = excluded.updated_at",
                (conversation_id, iso, iso),
            )
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS seq FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            seq = int(row["seq"])
            conn.executemany(
                "INSERT INTO messages (id, conversation_id, seq, created_at, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        item.id,
                        conversation_id,
                        seq + offset,
                        iso,
                        json.dumps(item.to_payload(), ensure_ascii=False),
                    )
                    for offset, item in enumerate(stored, start=1)
                ],
            )
            if self._max_messages:
                conn.execute(
                    "DELETE FROM messages WHERE conversation_id = ? AND seq <= ("
                    "  SELECT MAX(seq) - ? FROM messages WHERE conversation_id = ?"
                    ")",
                    (conversation_id, self._max_messages, conversation_id),
                )

    async def get(
        self, conversation_id: str, *, limit: int | None = None
    ) -> list[StoredMessage]:
        return await asyncio.to_thread(self._get_sync, conversation_id, limit)

    def _get_sync(
        self, conversation_id: str, limit: int | None
    ) -> list[StoredMessage]:
        with self._connect() as conn:
            if limit:
                rows = conn.execute(
                    "SELECT payload FROM messages WHERE conversation_id = ? "
                    "ORDER BY seq DESC LIMIT ?",
                    (conversation_id, limit),
                ).fetchall()
                rows = list(reversed(rows))
            else:
                rows = conn.execute(
                    "SELECT payload FROM messages WHERE conversation_id = ? ORDER BY seq",
                    (conversation_id,),
                ).fetchall()
        return [
            StoredMessage.from_payload(conversation_id, json.loads(row["payload"]))
            for row in rows
        ]

    async def meta(self, conversation_id: str) -> ConversationMeta | None:
        return await asyncio.to_thread(self._meta_sync, conversation_id)

    def _meta_sync(self, conversation_id: str) -> ConversationMeta | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT c.created_at, c.updated_at, "
                "  (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.conversation_id) AS n "
                "FROM conversations c WHERE c.conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return ConversationMeta(
            conversation_id=conversation_id,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            message_count=int(row["n"]),
        )

    async def delete(self, conversation_id: str) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self._delete_sync, conversation_id)

    def _delete_sync(self, conversation_id: str) -> bool:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
            )
            cur = conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,)
            )
            return cur.rowcount > 0

    async def health(self) -> None:
        await asyncio.to_thread(self._health_sync)

    def _health_sync(self) -> None:
        with self._connect() as conn:
            conn.execute("SELECT 1 FROM conversations LIMIT 1").fetchone()
