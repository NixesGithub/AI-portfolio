"""Capa de servicio: pega la memoria con el bucle del agente.

La API no sabe qué es un `BaseMessage` ni cómo se recorta un historial, y el
bucle no sabe que existe una base de datos. Todo el trabajo de coordinar las
dos cosas —cargar el hilo, correr el turno, persistirlo entero— pasa acá.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from ..config import Settings
from ..memory.base import Conversation, ConversationStore
from . import history as history_utils
from .locks import KeyedLocks
from .loop import AgentResult, SupportsToolCalling, agent_loop
from .prompts import SYSTEM_PROMPT

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChatTurn:
    conversation_id: str
    result: AgentResult


class ChatService:
    def __init__(
        self,
        llm: SupportsToolCalling,
        tools: list[BaseTool],
        store: ConversationStore,
        settings: Settings,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._store = store
        self._settings = settings
        self._locks = KeyedLocks()

    async def send(self, conversation_id: str, message: str) -> ChatTurn:
        """Procesa un mensaje dentro de un hilo y devuelve la respuesta."""
        # El lock cubre el ciclo entero (leer → razonar → escribir), no sólo la
        # escritura: si sólo cubriera el INSERT, dos pedidos simultáneos
        # igualmente habrían razonado sobre el mismo historial viejo.
        async with self._locks.acquire(conversation_id):
            stored = await self._store.history(conversation_id)
            user_message = HumanMessage(content=message)

            prompt = history_utils.prepare(
                SYSTEM_PROMPT,
                [*stored, user_message],
                self._settings.history_max_tokens,
            )

            result = await agent_loop(
                self._llm,
                self._tools,
                prompt,
                max_iterations=self._settings.max_iterations,
                tool_timeout=self._settings.tool_timeout_seconds,
            )

            # El turno se persiste en una sola operación: la pregunta y todo lo
            # que produjo entran juntos o no entra nada.
            await self._store.append(
                conversation_id, [user_message, *result.new_messages]
            )

        log.info(
            "turno completado",
            extra={
                "iterations": result.iterations,
                "tool_calls": len(result.steps),
                "stop_reason": result.stop_reason,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
        )
        return ChatTurn(conversation_id=conversation_id, result=result)

    async def conversation(self, conversation_id: str) -> Conversation:
        """Historial completo de un hilo.

        Raises:
            ConversationNotFound: si el id no existe.
        """
        return await self._store.load(conversation_id)
