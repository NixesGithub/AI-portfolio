"""Tests del bucle del agente.

Es la pieza que se pidió escribir a mano, así que es la que más cubro: no
alcanza con "responde bien", hay que fijar qué pasa en cada camino de error.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from app.agent.loop import agent_loop
from tests.conftest import ScriptedLLM, fake_quote, tool_call


async def test_responde_sin_usar_herramientas():
    llm = ScriptedLLM([AIMessage("Hola", usage_metadata={
        "input_tokens": 12, "output_tokens": 4, "total_tokens": 16})])

    result = await agent_loop(llm, [fake_quote], [HumanMessage("hola")])

    assert result.output == "Hola"
    assert result.iterations == 1
    assert result.stop_reason == "end_turn"
    assert result.steps == []
    assert (result.input_tokens, result.output_tokens) == (12, 4)
    # Las tools se ofrecen aunque no se usen.
    assert llm.bound_tools == ["fake_quote"]


async def test_ejecuta_la_herramienta_y_devuelve_el_resultado_al_modelo():
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[tool_call("fake_quote", {"symbol": "AAPL"}, "c1")]),
        AIMessage("AAPL cotiza a 100."),
    ])

    result = await agent_loop(llm, [fake_quote], [HumanMessage("precio de AAPL?")])

    assert result.output == "AAPL cotiza a 100."
    assert result.iterations == 2
    assert [m.type for m in result.new_messages] == ["ai", "tool", "ai"]

    # El ToolMessage tiene que citar el id de la llamada: sin eso la API del
    # modelo rechaza el turno siguiente.
    tool_message = result.new_messages[1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.tool_call_id == "c1"
    assert '"price":100' in tool_message.content

    # Y el modelo tiene que haber visto ese resultado en la segunda llamada.
    assert any(isinstance(m, ToolMessage) for m in llm.calls[1])


async def test_llamadas_de_una_misma_tanda_corren_en_paralelo():
    @tool
    async def lenta(symbol: str) -> str:
        """Tarda a propósito."""
        await asyncio.sleep(0.1)
        return symbol

    llm = ScriptedLLM([
        AIMessage("", tool_calls=[
            tool_call("lenta", {"symbol": "A"}, "c1"),
            tool_call("lenta", {"symbol": "B"}, "c2"),
            tool_call("lenta", {"symbol": "C"}, "c3"),
        ]),
        AIMessage("Listo."),
    ])

    started = asyncio.get_running_loop().time()
    result = await agent_loop(llm, [lenta], [HumanMessage("comparar A, B y C")])
    elapsed = asyncio.get_running_loop().time() - started

    assert len(result.steps) == 3
    # En serie serían ~300 ms. El margen es amplio para no volverlo frágil en CI.
    assert elapsed < 0.25


async def test_una_herramienta_que_falla_no_tumba_el_turno():
    @tool
    async def rota(symbol: str) -> str:
        """Siempre falla."""
        raise RuntimeError("el proveedor se cayó")

    llm = ScriptedLLM([
        AIMessage("", tool_calls=[tool_call("rota", {"symbol": "AAPL"}, "c1")]),
        AIMessage("No pude obtener ese dato ahora."),
    ])

    result = await agent_loop(llm, [rota], [HumanMessage("precio?")])

    assert result.output == "No pude obtener ese dato ahora."
    assert result.steps[0].ok is False
    assert result.steps[0].error == "RuntimeError"
    # El modelo se entera del fallo, pero nunca del traceback.
    content = result.new_messages[1].content
    assert '"code": "tool_error"' in content
    assert "el proveedor se cayó" not in content


async def test_herramienta_inexistente_le_avisa_al_modelo_cuales_hay():
    llm = ScriptedLLM([
        AIMessage("", tool_calls=[tool_call("inventada", {}, "c1")]),
        AIMessage("Uso la correcta."),
    ])

    result = await agent_loop(llm, [fake_quote], [HumanMessage("x")])

    assert result.steps[0].error == "unknown_tool"
    assert "fake_quote" in result.new_messages[1].content


async def test_timeout_de_herramienta():
    @tool
    async def eterna(symbol: str) -> str:
        """No termina nunca."""
        await asyncio.sleep(10)
        return "tarde"

    llm = ScriptedLLM([
        AIMessage("", tool_calls=[tool_call("eterna", {"symbol": "A"}, "c1")]),
        AIMessage("Se demoró demasiado."),
    ])

    result = await agent_loop(
        llm, [eterna], [HumanMessage("x")], tool_timeout=0.05
    )

    assert result.steps[0].error == "tool_timeout"
    assert '"code": "tool_timeout"' in result.new_messages[1].content


async def test_tope_de_iteraciones_fuerza_una_respuesta_final():
    """Un modelo que pide herramientas para siempre tiene que cortarse.

    Y el turno tiene que terminar igual en un AIMessage: si quedara terminado
    en un ToolMessage, la conversación quedaría en un estado que la API del
    modelo rechaza en el turno siguiente.
    """
    # Tres vueltas pidiendo herramientas, y recién después una respuesta: es lo
    # que el modelo devolvería en la llamada de cierre, que va sin tools.
    llm = ScriptedLLM(
        [AIMessage("", tool_calls=[tool_call("fake_quote", {"symbol": "A"}, f"c{i}")])
         for i in range(3)]
        + [AIMessage("Con lo que junté: cotiza a 100.")]
    )

    result = await agent_loop(
        llm, [fake_quote], [HumanMessage("x")], max_iterations=3
    )

    assert result.stop_reason == "max_iterations"
    assert result.iterations == 3
    assert len(result.steps) == 3
    assert result.new_messages[-1].type == "ai"
    assert result.output == "Con lo que junté: cotiza a 100."

    # La llamada de cierre va sin herramientas, para que no pueda pedir más.
    assert not result.new_messages[-1].tool_calls


async def test_si_el_cierre_forzado_falla_igual_hay_respuesta():
    class LLMQueRompeAlCerrar(ScriptedLLM):
        async def ainvoke(self, messages):
            if self.bound_tools is not None and not self.script:
                raise RuntimeError("el proveedor se cayó justo ahora")
            return await super().ainvoke(messages)

    llm = LLMQueRompeAlCerrar(
        [AIMessage("", tool_calls=[tool_call("fake_quote", {"symbol": "A"}, "c1")])]
    )

    result = await agent_loop(
        llm, [fake_quote], [HumanMessage("x")], max_iterations=1
    )

    assert result.stop_reason == "max_iterations"
    assert result.output  # nunca vacío
    assert result.new_messages[-1].type == "ai"


async def test_max_iterations_invalido():
    with pytest.raises(ValueError):
        await agent_loop(
            ScriptedLLM([]), [fake_quote], [HumanMessage("x")], max_iterations=0
        )
