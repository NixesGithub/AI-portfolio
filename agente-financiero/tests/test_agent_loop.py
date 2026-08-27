"""Tests del bucle del agente: es el corazón del servicio y su contrato es estricto."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from app.agent.loop import agent_loop


class ScriptedModel(BaseChatModel):
    """Modelo con las respuestas escritas de antemano."""

    script: list[AIMessage] = []
    default: str = "Listo."
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        message = self.script.pop(0) if self.script else AIMessage(content=self.default)
        return ChatResult(generations=[ChatGeneration(message=message)])


class EchoInput(BaseModel):
    value: str


class EchoTool(BaseTool):
    name: str = "echo"
    description: str = "Devuelve lo que le pasas."
    args_schema: type[BaseModel] = EchoInput

    def _run(self, value: str, **_: Any) -> str:
        return f"eco:{value}"


class BoomTool(BaseTool):
    name: str = "boom"
    description: str = "Siempre falla."
    args_schema: type[BaseModel] = EchoInput

    def _run(self, value: str, **_: Any) -> str:
        raise RuntimeError("se rompió")


class SlowTool(BaseTool):
    name: str = "slow"
    description: str = "Tarda demasiado."
    args_schema: type[BaseModel] = EchoInput

    def _run(self, value: str, **_: Any) -> str:  # pragma: no cover - no llega a terminar
        return "nunca"

    async def _arun(self, value: str, **_: Any) -> str:
        await asyncio.sleep(5)
        return "nunca"


def call(name: str, value: str = "hola", call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": {"value": value}, "id": call_id, "type": "tool_call"}
        ],
    )


async def test_una_vuelta_con_tool_y_respuesta_final():
    model = ScriptedModel(
        script=[call("echo"), AIMessage(content="El eco dijo hola.")], seen=[]
    )
    result = await agent_loop(
        llm=model,
        tools=[EchoTool()],
        messages=[HumanMessage(content="di hola")],
        max_iterations=4,
    )

    assert result.stop_reason == "final_answer"
    assert result.output_text == "El eco dijo hola."
    assert result.iterations == 2
    assert [step.tool for step in result.steps] == ["echo"]
    assert result.steps[0].ok is True

    # Invariante: AIMessage con tool_calls -> ToolMessage con el mismo id.
    kinds = [type(m).__name__ for m in result.new_messages]
    assert kinds == ["AIMessage", "ToolMessage", "AIMessage"]
    assert result.new_messages[1].tool_call_id == "c1"
    assert result.new_messages[1].content == "eco:hola"


async def test_varias_tools_en_la_misma_vuelta_conservan_el_orden():
    multi = AIMessage(
        content="",
        tool_calls=[
            {"name": "echo", "args": {"value": "a"}, "id": "c1", "type": "tool_call"},
            {"name": "echo", "args": {"value": "b"}, "id": "c2", "type": "tool_call"},
        ],
    )
    model = ScriptedModel(script=[multi, AIMessage(content="ok")], seen=[])
    result = await agent_loop(
        llm=model, tools=[EchoTool()], messages=[HumanMessage(content="dos")]
    )

    tool_messages = [m for m in result.new_messages if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2"]
    assert [m.content for m in tool_messages] == ["eco:a", "eco:b"]


async def test_una_tool_que_falla_no_tumba_el_turno():
    model = ScriptedModel(
        script=[call("boom"), AIMessage(content="No pude, lo siento.")], seen=[]
    )
    result = await agent_loop(
        llm=model, tools=[BoomTool()], messages=[HumanMessage(content="rompe")]
    )

    assert result.stop_reason == "final_answer"
    assert result.steps[0].ok is False
    assert "se rompió" in result.steps[0].error
    tool_message = result.new_messages[1]
    assert tool_message.status == "error"
    assert "Error ejecutando 'boom'" in tool_message.content


async def test_tool_que_se_pasa_de_tiempo_se_cancela():
    model = ScriptedModel(script=[call("slow"), AIMessage(content="tardaba")], seen=[])
    result = await agent_loop(
        llm=model,
        tools=[SlowTool()],
        messages=[HumanMessage(content="espera")],
        tool_timeout_s=0.05,
    )

    assert result.steps[0].error == "timeout"
    assert result.steps[0].ok is False
    assert result.stop_reason == "final_answer"


async def test_tool_inexistente_se_le_explica_al_modelo():
    model = ScriptedModel(script=[call("fantasma"), AIMessage(content="vale")], seen=[])
    result = await agent_loop(
        llm=model, tools=[EchoTool()], messages=[HumanMessage(content="?")]
    )

    assert result.steps[0].error == "unknown_tool"
    assert "no existe" in result.new_messages[1].content
    assert "echo" in result.new_messages[1].content


async def test_se_corta_al_agotar_iteraciones_y_aun_asi_responde():
    # El modelo pide tool eternamente: el bucle debe cortar y forzar respuesta.
    model = ScriptedModel(
        script=[call("echo", call_id=f"c{i}") for i in range(3)],
        default="Me quedé sin herramientas, esto es lo que tengo.",
        seen=[],
    )
    result = await agent_loop(
        llm=model,
        tools=[EchoTool()],
        messages=[HumanMessage(content="bucle")],
        max_iterations=3,
    )

    assert result.stop_reason == "max_iterations"
    assert result.iterations == 3
    assert len(result.steps) == 3
    assert result.output_text.startswith("Me quedé sin herramientas")
    # El empujón final es interno: no se persiste en el historial.
    assert all(m.type != "system" for m in result.new_messages)
    assert result.new_messages[-1].type == "ai"
    # La última llamada al modelo lleva el empujón para que cierre el turno.
    assert model.seen[-1][-1].type == "system"


async def test_suma_el_consumo_de_tokens():
    respuesta = AIMessage(
        content="hola",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    model = ScriptedModel(script=[respuesta], seen=[])
    result = await agent_loop(llm=model, tools=[], messages=[HumanMessage(content="hi")])

    assert result.usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


async def test_emite_eventos_de_observabilidad():
    eventos: list[str] = []

    async def on_event(name: str, payload: dict) -> None:
        eventos.append(name)

    model = ScriptedModel(script=[call("echo"), AIMessage(content="fin")], seen=[])
    await agent_loop(
        llm=model,
        tools=[EchoTool()],
        messages=[HumanMessage(content="x")],
        on_event=on_event,
    )

    assert eventos == ["llm_start", "llm_end", "tool_end", "llm_start", "llm_end"]


async def test_respuesta_vacia_tiene_texto_de_reserva():
    model = ScriptedModel(script=[AIMessage(content="   ")], seen=[])
    result = await agent_loop(llm=model, tools=[], messages=[HumanMessage(content="x")])
    assert result.output_text
    assert "reformular" in result.output_text
