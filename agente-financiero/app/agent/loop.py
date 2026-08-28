"""El bucle del agente, escrito a mano.

El enunciado pide definir explícitamente el ``agent_loop``, y además es la
decisión correcta para un servicio en producción: un ``AgentExecutor`` de caja
negra esconde justo lo que hay que controlar —cuántas vueltas se dan, qué pasa
cuando una tool tarda o revienta, qué se guarda en el historial, qué se mide—.

El ciclo es el clásico ReAct sobre *tool calling*:

    modelo → ¿pide tools? → ejecutarlas → devolver resultados → modelo → …

y termina cuando el modelo contesta sin pedir nada más, o cuando se agota el
presupuesto de vueltas (y entonces se le fuerza a responder con lo que tenga).

Invariantes que el bucle garantiza:

* Todo ``AIMessage`` con ``tool_calls`` va seguido de un ``ToolMessage`` por cada
  llamada, con su ``tool_call_id``. Si esto se rompe, la siguiente petición al
  proveedor falla con un 400 difícil de diagnosticar.
* Una tool que falla o que agota su tiempo **no** tumba la petición: se le
  devuelve el error al modelo, que puede reintentar, cambiar de argumentos o
  explicárselo al usuario.
* El bucle es puro respecto al almacenamiento: recibe mensajes y devuelve
  mensajes. Quién los persiste es problema del servicio.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

StopReason = Literal["final_answer", "max_iterations"]

FORCE_ANSWER_PROMPT = (
    "Has agotado el número de herramientas que puedes usar en este turno. "
    "Responde ahora al usuario con la información que ya tienes, di explícitamente "
    "qué no has podido comprobar y no inventes ningún dato."
)

EMPTY_ANSWER_FALLBACK = (
    "No he podido elaborar una respuesta para eso. ¿Puedes reformular la pregunta?"
)


@dataclass
class ToolStep:
    """Traza de una ejecución de tool. Sirve para depurar y para observabilidad."""

    tool: str
    args: dict[str, Any]
    tool_call_id: str
    ok: bool
    latency_ms: int
    error: str | None = None
    output_preview: str = ""


@dataclass
class AgentResult:
    output_text: str
    new_messages: list[BaseMessage] = field(default_factory=list)
    steps: list[ToolStep] = field(default_factory=list)
    iterations: int = 0
    stop_reason: StopReason = "final_answer"
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0


def _accumulate_usage(total: dict[str, int], message: AIMessage) -> None:
    usage = getattr(message, "usage_metadata", None) or {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def _preview(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


async def _execute_tool_call(
    call: dict[str, Any],
    tools_by_name: dict[str, BaseTool],
    timeout_s: float,
) -> tuple[ToolMessage, ToolStep]:
    """Ejecuta una llamada a tool y la convierte en ``ToolMessage`` pase lo que pase."""
    name = call.get("name", "")
    args = call.get("args") or {}
    call_id = call.get("id") or ""
    started = time.perf_counter()

    def finish(content: str, ok: bool, error: str | None = None):
        latency_ms = int((time.perf_counter() - started) * 1000)
        message = ToolMessage(
            content=content,
            tool_call_id=call_id,
            name=name,
            status="success" if ok else "error",
        )
        step = ToolStep(
            tool=name,
            args=args,
            tool_call_id=call_id,
            ok=ok,
            latency_ms=latency_ms,
            error=error,
            output_preview=_preview(content),
        )
        return message, step

    tool = tools_by_name.get(name)
    if tool is None:
        available = ", ".join(sorted(tools_by_name)) or "(ninguna)"
        return finish(
            f"Error: la herramienta '{name}' no existe. Disponibles: {available}.",
            ok=False,
            error="unknown_tool",
        )

    try:
        output = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning("tool timeout", extra={"tool": name, "timeout_s": timeout_s})
        return finish(
            f"Error: '{name}' tardó más de {timeout_s:g}s y se canceló. "
            "Puedes reintentar con menos datos o explicárselo al usuario.",
            ok=False,
            error="timeout",
        )
    except Exception as exc:  # noqa: BLE001 - una tool jamás debe tumbar el turno
        logger.exception("tool error", extra={"tool": name})
        return finish(
            f"Error ejecutando '{name}': {type(exc).__name__}: {exc}",
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False, default=str)
    return finish(output, ok=True)


async def agent_loop(
    *,
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    messages: Sequence[BaseMessage],
    max_iterations: int = 6,
    tool_timeout_s: float = 15.0,
    on_event: Callable[[str, dict[str, Any]], Awaitable[None] | None] | None = None,
) -> AgentResult:
    """Ejecuta un turno completo del agente.

    Args:
        llm: modelo de chat con soporte de *tool calling*.
        tools: herramientas disponibles en este turno.
        messages: conversación completa que se le pasa al modelo (system incluido).
        max_iterations: llamadas al modelo como mucho; acota coste y latencia.
        tool_timeout_s: tiempo máximo por ejecución de tool.
        on_event: callback opcional de observabilidad (``llm_start``, ``tool_end``…).

    Returns:
        ``AgentResult`` con el texto final, los mensajes nuevos que hay que
        persistir (en orden), la traza de tools y el consumo de tokens.
    """
    started = time.perf_counter()
    tools_by_name = {tool.name: tool for tool in tools}
    model = llm.bind_tools(list(tools)) if tools else llm

    conversation: list[BaseMessage] = list(messages)
    new_messages: list[BaseMessage] = []
    steps: list[ToolStep] = []
    usage: dict[str, int] = {}

    async def emit(event: str, payload: dict[str, Any]) -> None:
        if on_event is None:
            return
        result = on_event(event, payload)
        if asyncio.iscoroutine(result):
            await result

    iterations = 0
    for iterations in range(1, max_iterations + 1):
        await emit("llm_start", {"iteration": iterations})
        response = await model.ainvoke(conversation)
        if not isinstance(response, AIMessage):  # pragma: no cover - contrato de LC
            response = AIMessage(content=str(response))
        _accumulate_usage(usage, response)
        conversation.append(response)
        new_messages.append(response)

        tool_calls = list(response.tool_calls or [])
        await emit(
            "llm_end", {"iteration": iterations, "tool_calls": len(tool_calls)}
        )

        if not tool_calls:
            text = response.text.strip() or EMPTY_ANSWER_FALLBACK
            return AgentResult(
                output_text=text,
                new_messages=new_messages,
                steps=steps,
                iterations=iterations,
                stop_reason="final_answer",
                usage=usage,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        # Varias tools en la misma vuelta se ejecutan en paralelo, pero los
        # ToolMessage se añaden en el orden de las tool_calls: el proveedor exige
        # esa correspondencia.
        results = await asyncio.gather(
            *(
                _execute_tool_call(call, tools_by_name, tool_timeout_s)
                for call in tool_calls
            )
        )
        for message, step in results:
            conversation.append(message)
            new_messages.append(message)
            steps.append(step)
            await emit(
                "tool_end",
                {
                    "tool": step.tool,
                    "ok": step.ok,
                    "latency_ms": step.latency_ms,
                    "error": step.error,
                },
            )

    # Se agotaron las vueltas: una última llamada sin tools para que cierre el
    # turno. El empujón va aparte y no se persiste; sólo la respuesta.
    logger.warning("agent_loop agotó las iteraciones", extra={"iterations": iterations})
    await emit("max_iterations", {"iterations": iterations})
    final = await llm.ainvoke([*conversation, SystemMessage(content=FORCE_ANSWER_PROMPT)])
    if not isinstance(final, AIMessage):  # pragma: no cover
        final = AIMessage(content=str(final))
    _accumulate_usage(usage, final)
    new_messages.append(final)
    return AgentResult(
        output_text=final.text.strip() or EMPTY_ANSWER_FALLBACK,
        new_messages=new_messages,
        steps=steps,
        iterations=iterations,
        stop_reason="max_iterations",
        usage=usage,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
