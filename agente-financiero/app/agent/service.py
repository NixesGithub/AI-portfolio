"""Servicio de conversación: pega memoria, prompt y bucle del agente.

La capa HTTP no sabe nada de LangChain y el bucle no sabe nada de HTTP. Todo lo
que hay entre medias —cargar el hilo, recortar la ventana, serializar por
conversación, decidir qué se persiste— vive aquí.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from app.agent.history import build_prompt_messages
from app.agent.loop import AgentResult, ToolStep, agent_loop
from app.agent.prompts import system_prompt
from app.errors import AgentTimeout, ConversationNotFound, UpstreamError
from app.memory.base import ConversationMeta, ConversationStore, StoredMessage, new_id
from app.memory.locks import ConversationLocks

logger = logging.getLogger(__name__)


@dataclass
class ChatTurn:
    conversation_id: str
    message_id: str
    reply: str
    created_at: datetime
    is_new_conversation: bool
    steps: list[ToolStep] = field(default_factory=list)
    iterations: int = 0
    stop_reason: str = "final_answer"
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0


class ChatService:
    def __init__(
        self,
        *,
        settings,
        store: ConversationStore,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        locks: ConversationLocks | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._llm = llm
        self._tools = list(tools)
        self._locks = locks or ConversationLocks()

    @property
    def store(self) -> ConversationStore:
        return self._store

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self._tools]

    async def send_message(
        self,
        *,
        message: str,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatTurn:
        conversation_id = conversation_id or new_id("conv")

        # Un hilo, un turno a la vez: dos peticiones simultáneas al mismo id
        # dejarían el historial intercalado e inconsistente.
        async with self._locks.acquire(conversation_id):
            history = await self._store.get(conversation_id)
            is_new = not history

            human = HumanMessage(content=message, additional_kwargs=metadata or {})
            prompt_messages = build_prompt_messages(
                system_prompt(), history, self._settings.history_window
            )
            prompt_messages.append(human)

            result = await self._run_agent(prompt_messages)

            # Sólo se persiste si el turno terminó bien: si falla, el usuario puede
            # reintentar sin que el hilo quede con un mensaje suyo sin respuesta.
            stored = await self._store.append(
                conversation_id, [human, *result.new_messages]
            )

        logger.info(
            "turno completado",
            extra={
                "conversation_id": conversation_id,
                "iterations": result.iterations,
                "stop_reason": result.stop_reason,
                "tool_calls": len(result.steps),
                "latency_ms": result.latency_ms,
                "tokens": result.usage.get("total_tokens"),
            },
        )
        return ChatTurn(
            conversation_id=conversation_id,
            message_id=stored[-1].id,
            reply=result.output_text,
            created_at=stored[-1].created_at,
            is_new_conversation=is_new,
            steps=result.steps,
            iterations=result.iterations,
            stop_reason=result.stop_reason,
            usage=result.usage,
            latency_ms=result.latency_ms,
        )

    async def _run_agent(self, prompt_messages: list) -> AgentResult:
        try:
            return await asyncio.wait_for(
                agent_loop(
                    llm=self._llm,
                    tools=self._tools,
                    messages=prompt_messages,
                    max_iterations=self._settings.agent_max_iterations,
                    tool_timeout_s=self._settings.tool_timeout_s,
                ),
                timeout=self._settings.agent_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise AgentTimeout(
                "El agente tardó demasiado en responder. Vuelve a intentarlo."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - el SDK del proveedor no tipa sus fallos
            logger.exception("fallo del proveedor del modelo")
            raise UpstreamError(
                f"El modelo no está disponible ahora mismo ({type(exc).__name__})."
            ) from exc

    async def get_conversation(
        self, conversation_id: str, *, limit: int | None = None
    ) -> tuple[ConversationMeta, list[StoredMessage]]:
        meta = await self._store.meta(conversation_id)
        if meta is None:
            raise ConversationNotFound(
                f"No existe la conversación '{conversation_id}'."
            )
        messages = await self._store.get(conversation_id, limit=limit)
        return meta, messages

    async def delete_conversation(self, conversation_id: str) -> None:
        if not await self._store.delete(conversation_id):
            raise ConversationNotFound(
                f"No existe la conversación '{conversation_id}'."
            )
