"""Fixtures compartidas.

Ningún test toca la red ni gasta tokens: el modelo es un doble que devuelve un
guion prefijado. Eso es lo que permite testear el bucle del agente —incluido
lo que pasa cuando una herramienta falla o cuando se agotan las iteraciones—
de forma determinista.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import pytest
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import BaseTool, tool

from app.agent.service import ChatService
from app.config import Settings
from app.memory import InMemoryConversationStore

logging.disable(logging.CRITICAL)


class ScriptedLLM:
    """Modelo falso que responde según un guion.

    Registra con qué mensajes lo llamaron, que es lo que hace falta para
    verificar que la memoria por conversación llega efectivamente al prompt.
    """

    def __init__(self, script: Sequence[AIMessage]) -> None:
        self.script = list(script)
        self.calls: list[list[AnyMessage]] = []
        self.bound_tools: list[str] | None = None

    def bind_tools(self, tools: Sequence[BaseTool]) -> "ScriptedLLM":
        self.bound_tools = [t.name for t in tools]
        return self

    async def ainvoke(self, messages: Sequence[AnyMessage]) -> AIMessage:
        self.calls.append(list(messages))
        if not self.script:
            return AIMessage(content="(guion agotado)")
        return self.script.pop(0)


def tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


@tool
async def fake_quote(symbol: str) -> str:
    """Cotización de prueba."""
    return f'{{"symbol":"{symbol}","price":100}}'


@pytest.fixture
def settings() -> Settings:
    return Settings(
        anthropic_api_key="test-key",
        database_url="sqlite:///./data/test.db",
        max_iterations=3,
        tool_timeout_seconds=1.0,
    )


@pytest.fixture
def store() -> InMemoryConversationStore:
    return InMemoryConversationStore()


@pytest.fixture
def make_service(store, settings):
    def _make(script: Sequence[AIMessage], tools: list[BaseTool] | None = None):
        llm = ScriptedLLM(script)
        service = ChatService(llm, tools or [fake_quote], store, settings)
        return service, llm

    return _make
