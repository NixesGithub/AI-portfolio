"""Ventana de contexto: qué parte del historial ve el modelo.

Una conversación crece sin límite, pero el contexto (y el coste por token) no. Se
le pasan al modelo los últimos ``window`` mensajes, con una condición que no es
opcional: la ventana no puede empezar por un ``ToolMessage`` huérfano. Los
proveedores rechazan con un 400 cualquier resultado de tool cuya llamada no esté
en la conversación, y es un error que sólo aparece en hilos largos —justo en
producción y no en las pruebas—.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage

from app.memory.base import StoredMessage


def trim_history(messages: Sequence[BaseMessage], window: int) -> list[BaseMessage]:
    """Últimos ``window`` mensajes, sin dejar ``ToolMessage`` sin su llamada."""
    if window <= 0 or len(messages) <= window:
        start = 0
    else:
        start = len(messages) - window
    # Si el corte cae entre un AIMessage con tool_calls y sus resultados,
    # descartamos los resultados huérfanos.
    while start < len(messages) and isinstance(messages[start], ToolMessage):
        start += 1
    return list(messages[start:])


def build_prompt_messages(
    system: str,
    history: Iterable[StoredMessage],
    window: int,
) -> list[BaseMessage]:
    """Construye la entrada del modelo: system + ventana del historial."""
    stored = [item.message for item in history]
    return [SystemMessage(content=system), *trim_history(stored, window)]
