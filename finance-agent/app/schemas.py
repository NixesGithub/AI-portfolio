"""Contratos HTTP.

Separados de los modelos internos a propósito: `MessageRecord` lleva un
`BaseMessage` de LangChain con bloques de contenido y metadata del proveedor,
y nada de eso tiene por qué formar parte de la API pública. Si mañana se
cambia de framework de agentes, estos esquemas no se mueven.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["user", "assistant", "tool", "system"]

# Ids controlados: entran en claves de caché, en nombres de lock y en consultas.
# Aceptar cualquier string es pedir que alguien mande 10 KB de basura como id.
CONVERSATION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"conversation_id": "demo-1", "message": "¿A cuánto cotiza Apple?"}
            ]
        }
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Mensaje del usuario.",
    )
    conversation_id: str | None = Field(
        default=None,
        pattern=CONVERSATION_ID_PATTERN,
        description=(
            "Hilo al que pertenece el mensaje. Si se omite, se crea uno nuevo "
            "y su id viene en la respuesta."
        ),
    )


class ToolStep(BaseModel):
    """Una llamada a herramienta, expuesta para que el cliente pueda auditar
    de dónde salió cada número."""

    iteration: int
    tool: str
    arguments: dict[str, Any]
    ok: bool
    duration_ms: int
    error: str | None = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    iterations: int = Field(description="Vueltas del bucle del agente.")
    stop_reason: Literal["end_turn", "max_iterations"]
    steps: list[ToolStep]
    usage: Usage


class Message(BaseModel):
    seq: int
    role: Role
    content: str
    created_at: datetime
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ConversationResponse(BaseModel):
    conversation_id: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    messages: list[Message]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    environment: str
    model: str


class ErrorResponse(BaseModel):
    detail: str
    code: str
    request_id: str
