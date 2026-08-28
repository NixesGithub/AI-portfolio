"""Contrato de la memoria conversacional.

El agente no sabe dónde se guardan los mensajes: habla con esta interfaz. Cambiar
de proceso a SQLite (o mañana a Postgres/Redis) es cambiar la implementación, no
el agente.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


@dataclass(frozen=True)
class StoredMessage:
    """Un mensaje de LangChain más los metadatos que necesita la API."""

    id: str
    conversation_id: str
    created_at: datetime
    message: BaseMessage

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "message": messages_to_dict([self.message])[0],
        }

    @classmethod
    def from_payload(
        cls, conversation_id: str, payload: dict[str, Any]
    ) -> "StoredMessage":
        return cls(
            id=payload["id"],
            conversation_id=conversation_id,
            created_at=datetime.fromisoformat(payload["created_at"]),
            message=messages_from_dict([payload["message"]])[0],
        )


@dataclass(frozen=True)
class ConversationMeta:
    conversation_id: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationStore(ABC):
    """Historial por ``conversation_id``."""

    @abstractmethod
    async def append(
        self, conversation_id: str, messages: Sequence[BaseMessage]
    ) -> list[StoredMessage]:
        """Añade mensajes al final del hilo y devuelve lo que se guardó."""

    @abstractmethod
    async def get(
        self, conversation_id: str, *, limit: int | None = None
    ) -> list[StoredMessage]:
        """Historial en orden cronológico. ``limit`` devuelve los N más recientes."""

    @abstractmethod
    async def meta(self, conversation_id: str) -> ConversationMeta | None:
        """Metadatos, o ``None`` si la conversación no existe."""

    @abstractmethod
    async def delete(self, conversation_id: str) -> bool:
        """Borra el hilo. ``True`` si existía."""

    async def exists(self, conversation_id: str) -> bool:
        return await self.meta(conversation_id) is not None

    async def health(self) -> None:
        """Lanza excepción si el backend no está operativo."""

    async def close(self) -> None:
        """Libera recursos al apagar el servicio."""
