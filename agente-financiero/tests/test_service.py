"""Tests del servicio: concurrencia por hilo y traducción de fallos."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent.models import FakeChatModel
from app.agent.service import ChatService
from app.errors import AgentTimeout, ConversationNotFound, UpstreamError
from app.memory import ConversationLocks, InMemoryConversationStore


class SlowModel(BaseChatModel):
    delay: float = 0.05

    @property
    def _llm_type(self) -> str:
        return "slow"

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        import time

        time.sleep(self.delay)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


class BrokenModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "broken"

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise RuntimeError("429 rate limit")


def build_service(settings, llm=None, store=None) -> ChatService:
    return ChatService(
        settings=settings,
        store=store or InMemoryConversationStore(),
        llm=llm or FakeChatModel(),
        tools=[],
        locks=ConversationLocks(),
    )


async def test_los_turnos_del_mismo_hilo_se_serializan(settings):
    service = build_service(settings, llm=SlowModel())

    await asyncio.gather(
        *(service.send_message(message=f"mensaje {i}", conversation_id="c1") for i in range(5))
    )

    history = await service.store.get("c1")
    tipos = [item.message.type for item in history]
    assert tipos == ["human", "ai"] * 5  # ni un solo turno intercalado


async def test_hilos_distintos_van_en_paralelo(settings):
    service = build_service(settings, llm=SlowModel(delay=0.1))

    started = asyncio.get_event_loop().time()
    await asyncio.gather(
        *(service.send_message(message="hola", conversation_id=f"c{i}") for i in range(5))
    )
    elapsed = asyncio.get_event_loop().time() - started

    # En serie serían ~0.5 s; el cerrojo es por conversación, no global.
    assert elapsed < 0.4


async def test_el_timeout_del_agente_se_traduce(settings):
    service = build_service(
        settings.model_copy(update={"agent_timeout_s": 0.01}), llm=SlowModel(delay=0.5)
    )

    with pytest.raises(AgentTimeout):
        await service.send_message(message="hola", conversation_id="c1")


async def test_un_fallo_del_proveedor_se_traduce(settings):
    service = build_service(settings, llm=BrokenModel())

    with pytest.raises(UpstreamError):
        await service.send_message(message="hola", conversation_id="c1")


async def test_un_turno_fallido_no_ensucia_el_historial(settings):
    service = build_service(settings, llm=BrokenModel())

    with pytest.raises(UpstreamError):
        await service.send_message(message="hola", conversation_id="c1")

    # El usuario puede reintentar sin que le quede un mensaje sin respuesta.
    assert await service.store.get("c1") == []
    assert await service.store.exists("c1") is False


async def test_historial_de_un_hilo_inexistente(settings):
    service = build_service(settings)

    with pytest.raises(ConversationNotFound):
        await service.get_conversation("fantasma")
