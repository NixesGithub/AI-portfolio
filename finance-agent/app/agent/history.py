"""Recorte del historial que se le manda al modelo.

Una conversación no tiene por qué caber en la ventana de contexto, y aunque
quepa, mandarla entera se paga por token en cada turno. Recortamos por tokens
y no por cantidad de mensajes porque un resultado de `get_price_history` pesa
lo que veinte turnos de charla.

El detalle que importa: no se puede cortar en cualquier lado. Un ToolMessage
huérfano (sin el AIMessage que lo pidió) hace que la API devuelva un 400. Por
eso `start_on="human"`, que garantiza que el corte cae en un límite de turno.
"""

from __future__ import annotations

from langchain_core.messages import AnyMessage, SystemMessage, trim_messages


def prepare(
    system_prompt: str, history: list[AnyMessage], max_tokens: int
) -> list[AnyMessage]:
    """Devuelve el system prompt seguido del historial recortado."""
    trimmed = trim_messages(
        history,
        max_tokens=max_tokens,
        # El contador aproximado evita una llamada de red por turno sólo para
        # contar. Sobreestima un poco, que es el lado seguro del error.
        token_counter="approximate",
        strategy="last",
        start_on="human",
        include_system=False,
        allow_partial=False,
    )
    return [SystemMessage(content=system_prompt), *trimmed]
