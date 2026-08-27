"""Contratos de entrada y salida de la API.

Pydantic hace de frontera: lo que no valida aquí no llega al agente. También es
la documentación viva del servicio, porque de estos modelos sale el OpenAPI que
se sirve en ``/docs``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from app.agent.loop import ToolStep
from app.agent.service import ChatTurn
from app.memory.base import ConversationMeta, StoredMessage

CONVERSATION_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,64}$"

_ROLES: dict[str, str] = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
}


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=4000,
        description="Mensaje del usuario.",
        examples=["¿Cómo va Apple hoy?"],
    )
    conversation_id: str | None = Field(
        default=None,
        pattern=CONVERSATION_ID_PATTERN,
        description=(
            "Hilo al que pertenece el mensaje. Si se omite, se crea uno nuevo y su "
            "id viene en la respuesta. El cliente puede imponer el suyo."
        ),
        examples=["conv_demo_1"],
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Datos libres del cliente que se guardan junto al mensaje.",
    )


class ToolCallModel(BaseModel):
    id: str | None = None
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolStepModel(BaseModel):
    tool: str
    args: dict[str, Any]
    ok: bool
    latency_ms: int
    error: str | None = None

    @classmethod
    def from_step(cls, step: ToolStep) -> "ToolStepModel":
        return cls(
            tool=step.tool,
            args=step.args,
            ok=step.ok,
            latency_ms=step.latency_ms,
            error=step.error,
        )


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    reply: str
    created_at: datetime
    is_new_conversation: bool
    stop_reason: Literal["final_answer", "max_iterations"]
    iterations: int
    latency_ms: int
    tool_steps: list[ToolStepModel] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def from_turn(cls, turn: ChatTurn) -> "ChatResponse":
        return cls(
            conversation_id=turn.conversation_id,
            message_id=turn.message_id,
            reply=turn.reply,
            created_at=turn.created_at,
            is_new_conversation=turn.is_new_conversation,
            stop_reason=turn.stop_reason,  # type: ignore[arg-type]
            iterations=turn.iterations,
            latency_ms=turn.latency_ms,
            tool_steps=[ToolStepModel.from_step(step) for step in turn.steps],
            usage=turn.usage,
        )


class MessageModel(BaseModel):
    id: str
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    created_at: datetime
    tool_calls: list[ToolCallModel] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None

    @classmethod
    def from_stored(cls, stored: StoredMessage) -> "MessageModel":
        message: BaseMessage = stored.message
        tool_calls = [
            ToolCallModel(id=call.get("id"), name=call["name"], args=call.get("args") or {})
            for call in (getattr(message, "tool_calls", None) or [])
        ]
        return cls(
            id=stored.id,
            role=_ROLES.get(message.type, message.type),  # type: ignore[arg-type]
            content=message.text,
            created_at=stored.created_at,
            tool_calls=tool_calls or None,
            tool_call_id=getattr(message, "tool_call_id", None),
            tool_name=getattr(message, "name", None),
        )


class ConversationResponse(BaseModel):
    conversation_id: str
    created_at: datetime
    updated_at: datetime
    message_count: int = Field(description="Mensajes guardados en total, sin filtrar.")
    messages: list[MessageModel]

    @classmethod
    def build(
        cls, meta: ConversationMeta, messages: list[MessageModel]
    ) -> "ConversationResponse":
        return cls(
            conversation_id=meta.conversation_id,
            created_at=meta.created_at,
            updated_at=meta.updated_at,
            message_count=meta.message_count,
            messages=messages,
        )


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    environment: str
    llm_provider: str
    store_backend: str
    tools: list[str]
