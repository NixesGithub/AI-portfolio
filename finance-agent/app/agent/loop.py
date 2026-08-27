"""El bucle del agente, escrito a mano.

Se pide explícitamente no delegarlo en `AgentExecutor` ni en un grafo armado.
El bucle es el mismo que implementa cualquier framework:

    1. Se le manda al modelo la conversación y la lista de herramientas.
    2. Si la respuesta no pide herramientas, ese texto es la respuesta final.
    3. Si las pide, se ejecutan, se agregan los resultados como ToolMessage y
       se vuelve al paso 1.

Lo que un framework esconde y acá está a la vista es todo lo demás: el tope de
iteraciones, qué pasa cuando una herramienta falla, qué pasa cuando el modelo
se queda sin vueltas, y qué se registra de cada paso.

Decisiones que vale la pena señalar:

- **Los errores de herramienta vuelven al modelo, no al usuario.** Un ticker
  mal escrito llega como ToolMessage con `{"error": ...}` y el modelo tiene la
  chance de corregirse. Sólo se propaga hacia arriba lo que impide seguir.
- **Las llamadas de una misma tanda corren en paralelo.** Cuando el modelo pide
  dos cotizaciones para comparar, tarda lo que la más lenta y no la suma.
- **Al agotarse las iteraciones se fuerza un cierre.** Se hace una última
  llamada sin herramientas para que el usuario reciba una respuesta en prosa y
  el historial no quede terminado en un ToolMessage, que es un estado que la
  API rechaza en el turno siguiente.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import BaseTool

log = logging.getLogger(__name__)

StopReason = Literal["end_turn", "max_iterations"]


class SupportsToolCalling(Protocol):
    """Lo mínimo que el bucle le pide al modelo.

    Tipar contra esto y no contra `ChatAnthropic` es lo que permite que los
    tests corran con un modelo falso sin tocar la red ni parchear imports.
    """

    def bind_tools(self, tools: Sequence[BaseTool]) -> Any: ...

    async def ainvoke(self, messages: Sequence[AnyMessage]) -> AIMessage: ...


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """Una llamada a herramienta, para la traza."""

    iteration: int
    tool: str
    args: dict[str, Any]
    ok: bool
    duration_ms: int
    error: str | None = None


@dataclass(slots=True)
class AgentResult:
    output: str
    # Sólo los mensajes nuevos del turno: el llamador ya tiene el historial
    # previo y los persiste todos juntos.
    new_messages: list[AnyMessage] = field(default_factory=list)
    steps: list[ToolInvocation] = field(default_factory=list)
    iterations: int = 0
    stop_reason: StopReason = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0


_NO_ANSWER = (
    "No pude terminar de resolver la consulta: agoté los intentos de búsqueda "
    "de datos. Probá acotando la pregunta a un símbolo por vez."
)


async def _execute_tool_call(
    call: dict[str, Any],
    tools_by_name: dict[str, BaseTool],
    iteration: int,
    timeout: float,
) -> tuple[ToolMessage, ToolInvocation]:
    """Ejecuta una llamada y la traduce a ToolMessage, pase lo que pase."""
    name = call.get("name", "")
    args = call.get("args") or {}
    call_id = call.get("id") or ""
    started = time.perf_counter()

    def _finish(content: str, ok: bool, error: str | None = None):
        elapsed = int((time.perf_counter() - started) * 1000)
        return (
            ToolMessage(content=content, tool_call_id=call_id, name=name),
            ToolInvocation(
                iteration=iteration, tool=name, args=args, ok=ok,
                duration_ms=elapsed, error=error,
            ),
        )

    tool = tools_by_name.get(name)
    if tool is None:
        # El modelo alucinó una herramienta. Decírselo suele bastar para que
        # use una de las que existen en la vuelta siguiente.
        log.warning("el modelo pidió una tool inexistente", extra={"tool": name})
        return _finish(
            json.dumps({
                "error": f"La herramienta {name!r} no existe.",
                "code": "unknown_tool",
                "available": sorted(tools_by_name),
            }, ensure_ascii=False),
            ok=False,
            error="unknown_tool",
        )

    try:
        result = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)
        return _finish(result if isinstance(result, str) else json.dumps(
            result, ensure_ascii=False, default=str), ok=True)
    except asyncio.TimeoutError:
        log.warning("timeout ejecutando tool", extra={"tool": name, "timeout": timeout})
        return _finish(
            json.dumps({
                "error": f"La herramienta {name!r} no respondió a tiempo.",
                "code": "tool_timeout",
            }, ensure_ascii=False),
            ok=False,
            error="tool_timeout",
        )
    except Exception as exc:
        # Una tool que rompe es un bug nuestro: se loguea entero. Al modelo le
        # llega el tipo de error, nunca el traceback.
        # `args` es un atributo reservado de LogRecord: usarlo como extra hace
        # que logging levante KeyError y se coma el log del error original.
        log.exception("tool falló", extra={"tool": name, "tool_args": args})
        return _finish(
            json.dumps({
                "error": f"La herramienta {name!r} falló ({type(exc).__name__}).",
                "code": "tool_error",
            }, ensure_ascii=False),
            ok=False,
            error=type(exc).__name__,
        )


def _accumulate_usage(result: AgentResult, message: AIMessage) -> None:
    usage = getattr(message, "usage_metadata", None) or {}
    result.input_tokens += usage.get("input_tokens", 0) or 0
    result.output_tokens += usage.get("output_tokens", 0) or 0


async def agent_loop(
    llm: SupportsToolCalling,
    tools: Sequence[BaseTool],
    messages: Sequence[AnyMessage],
    *,
    max_iterations: int = 6,
    tool_timeout: float = 20.0,
) -> AgentResult:
    """Corre el ciclo razonar → usar herramientas → responder.

    Args:
        llm: Modelo con capacidad de tool calling.
        tools: Herramientas disponibles en este turno.
        messages: Conversación completa a mandar (system prompt incluido).
        max_iterations: Tope de vueltas con llamadas a herramientas.
        tool_timeout: Techo por llamada individual.

    Returns:
        AgentResult con el texto final, los mensajes nuevos del turno y la
        traza de lo que se ejecutó.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations tiene que ser >= 1")

    tools_by_name = {tool.name: tool for tool in tools}
    model = llm.bind_tools(tools)

    conversation: list[AnyMessage] = list(messages)
    result = AgentResult(output="")

    for iteration in range(1, max_iterations + 1):
        result.iterations = iteration
        response: AIMessage = await model.ainvoke(conversation)
        conversation.append(response)
        result.new_messages.append(response)
        _accumulate_usage(result, response)

        if not response.tool_calls:
            result.output = response.text
            result.stop_reason = "end_turn"
            log.info(
                "turno resuelto",
                extra={"iterations": iteration, "tool_calls": len(result.steps)},
            )
            return result

        log.info(
            "el modelo pidió herramientas",
            extra={
                "iteration": iteration,
                "tools": [c.get("name") for c in response.tool_calls],
            },
        )

        executed = await asyncio.gather(*(
            _execute_tool_call(call, tools_by_name, iteration, tool_timeout)
            for call in response.tool_calls
        ))
        for tool_message, invocation in executed:
            conversation.append(tool_message)
            result.new_messages.append(tool_message)
            result.steps.append(invocation)

    # Sin vueltas disponibles. Una última llamada sin herramientas obliga al
    # modelo a responder con lo que ya juntó, en vez de dejar al usuario sin
    # nada y el historial terminado en un ToolMessage.
    log.warning(
        "se agotaron las iteraciones; forzando respuesta final",
        extra={"max_iterations": max_iterations},
    )
    result.stop_reason = "max_iterations"
    try:
        closing: AIMessage = await llm.ainvoke(conversation)
        conversation.append(closing)
        result.new_messages.append(closing)
        _accumulate_usage(result, closing)
        result.output = closing.text or _NO_ANSWER
    except Exception:
        log.exception("falló la respuesta de cierre")
        fallback = AIMessage(content=_NO_ANSWER)
        result.new_messages.append(fallback)
        result.output = _NO_ANSWER

    return result
