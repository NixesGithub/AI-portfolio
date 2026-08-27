"""Contrato de la memoria conversacional.

El resto de la aplicación depende de este protocolo y nunca de SQLite. Cuando
haga falta correr varias réplicas detrás de un balanceador, se escribe un
`PostgresConversationStore` y no se toca ni el agente ni la API.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence, runtime_checkable

from langchain_core.messages import BaseMessage


class ConversationNotFound(LookupError):
    """No existe una conversación con ese id."""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(f"conversation {conversation_id!r} not found")
        self.conversation_id = conversation_id


@dataclass(frozen=True, slots=True)
class MessageRecord:
    """Un mensaje tal como quedó guardado.

    `seq` es el orden dentro de la conversación: es lo que hace que el
    historial sea reproducible aunque dos mensajes caigan en el mismo
    milisegundo.
    """

    seq: int
    message: BaseMessage
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Conversation:
    id: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageRecord]


@runtime_checkable
class ConversationStore(Protocol):
    """Persistencia de hilos de conversación indexados por id."""

    async def load(self, conversation_id: str) -> Conversation:
        """Devuelve la conversación completa.

        Raises:
            ConversationNotFound: si el id no existe.
        """
        ...

    async def history(self, conversation_id: str) -> list[BaseMessage]:
        """Mensajes de un hilo, en orden. Lista vacía si el hilo no existe.

        A diferencia de `load`, esto no falla para un id nuevo: mandar el
        primer mensaje de una conversación es el caso normal, no un error.
        """
        ...

    async def append(
        self, conversation_id: str, messages: Sequence[BaseMessage]
    ) -> None:
        """Agrega mensajes al final del hilo, creándolo si no existía.

        Es atómico: o entran todos los mensajes del turno o no entra ninguno.
        Un turno a medias (la pregunta sin la respuesta, o una llamada a tool
        sin su resultado) deja el historial en un estado que el modelo rechaza
        en la request siguiente.
        """
        ...
