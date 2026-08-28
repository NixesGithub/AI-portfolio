"""Fábrica del modelo de chat.

Dos proveedores:

* ``anthropic``: el real, vía ``langchain-anthropic``.
* ``fake``: un modelo determinista, sin red ni credenciales, que sí usa la tool.
  Existe por dos razones muy prácticas: los tests corren en CI sin claves, y
  quien revise esto puede levantar la API y ver el circuito completo sin tener
  cuenta de Anthropic.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool


class ConfigurationError(RuntimeError):
    pass


# --- modelo falso ----------------------------------------------------------

_TICKER_RE = re.compile(r"\b([A-Z]{1,5}(?:[.\-][A-Z]{1,3})?|\^[A-Z]{2,6})\b")
_ALIASES = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "google": "GOOGL",
    "nvidia": "NVDA",
    "santander": "SAN.MC",
    "bitcoin": "BTC-USD",
}
_STOPWORDS = {"Y", "O", "A", "DE", "EL", "LA", "QUE", "OK", "IA", "API"}


def _guess_symbol(text: str) -> str | None:
    lowered = text.lower()
    for alias, symbol in _ALIASES.items():
        if alias in lowered:
            return symbol
    for candidate in _TICKER_RE.findall(text):
        if candidate.upper() not in _STOPWORDS:
            return candidate
    return None


class FakeChatModel(BaseChatModel):
    """Modelo determinista que emula *tool calling*.

    Si el último turno del usuario menciona un valor y todavía no hay resultado de
    la tool, pide ``yahoo_finance``. Si ya lo hay, redacta la respuesta con esos
    datos. En cualquier otro caso responde recordando el historial, que es lo que
    interesa comprobar sin gastar tokens.
    """

    tool_names: list[str] = []
    call_counter: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def bind_tools(  # type: ignore[override]
        self, tools: Sequence[Any], **kwargs: Any
    ) -> Runnable:
        names = [t.name if isinstance(t, BaseTool) else str(t) for t in tools]
        return self.model_copy(update={"tool_names": names})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        humans = [m for m in messages if isinstance(m, HumanMessage)]
        last_human = humans[-1].text if humans else ""

        if tool_messages:
            payloads = []
            for message in tool_messages[-3:]:
                try:
                    payloads.append(json.loads(message.text))
                except (json.JSONDecodeError, TypeError):
                    payloads.append({"raw": message.text})
            text = _render_fake_answer(payloads)
            return _result(AIMessage(content=text))

        symbol = _guess_symbol(last_human) if "yahoo_finance" in self.tool_names else None
        if symbol:
            self.call_counter += 1
            return _result(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "yahoo_finance",
                            "args": {"symbol": symbol, "section": "quote"},
                            "id": f"call_fake_{self.call_counter}",
                            "type": "tool_call",
                        }
                    ],
                )
            )

        previous = [m.text for m in humans[:-1]]
        recordatorio = (
            f" Antes me dijiste: {previous[-1]!r}." if previous else ""
        )
        return _result(
            AIMessage(
                content=(
                    f"[modo fake] Llevo {len(humans)} mensaje(s) tuyos en este hilo."
                    f"{recordatorio} Pregúntame por un valor (AAPL, MSFT, BTC-USD) "
                    "para que consulte Yahoo Finance, o arranca con "
                    "LLM_PROVIDER=anthropic para hablar con el modelo real."
                )
            )
        )


def _render_fake_answer(payloads: list[dict[str, Any]]) -> str:
    lines = []
    for payload in payloads:
        if "raw" in payload:  # la tool devolvió texto, no JSON (p. ej. un timeout)
            lines.append(str(payload["raw"]))
            continue
        if payload.get("error"):
            lines.append(f"No pude consultarlo: {payload.get('message')}")
            continue
        symbol = payload.get("symbol", "?")
        if payload.get("section") == "quote":
            lines.append(
                f"{symbol}: {payload.get('last_price')} {payload.get('currency') or ''}"
                f" ({payload.get('change_pct')}% frente al cierre anterior)."
            )
        elif payload.get("section") == "history":
            lines.append(
                f"{symbol}: de {payload.get('first_close')} a {payload.get('last_close')}"
                f" en {payload.get('period')} ({payload.get('change_pct')}%)."
            )
        else:
            lines.append(f"{symbol}: {payload.get('name')} — {payload.get('sector')}.")
    lines.append("[modo fake] Datos reales de Yahoo Finance, redacción simulada.")
    return "\n".join(lines)


def _result(message: AIMessage) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=message)])


# --- fábrica ---------------------------------------------------------------


def build_llm(settings) -> BaseChatModel:
    if settings.llm_provider == "fake":
        return FakeChatModel()

    if not settings.anthropic_api_key:
        raise ConfigurationError(
            "Falta ANTHROPIC_API_KEY. Define la clave o arranca con LLM_PROVIDER=fake."
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_s,
        # Reintentos con backoff ante 429/5xx del proveedor: los cortes puntuales
        # no deberían llegar al usuario.
        max_retries=settings.llm_max_retries,
        api_key=settings.anthropic_api_key,
    )
