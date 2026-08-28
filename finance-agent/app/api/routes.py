"""Endpoints HTTP."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from ..agent.service import ChatService
from ..config import Settings, get_settings
from ..logging_conf import conversation_id_var
from ..memory.base import Conversation, ConversationNotFound
from ..schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    ErrorResponse,
    HealthResponse,
    Message,
    Role,
    ToolStep,
    Usage,
    CONVERSATION_ID_PATTERN,
)
from .deps import get_chat_service

log = logging.getLogger(__name__)

router = APIRouter()

_ROLES: dict[str, Role] = {
    "human": "user",
    "ai": "assistant",
    "tool": "tool",
    "system": "system",
}


def _to_public(seq: int, message: BaseMessage, created_at) -> Message:
    """Traduce un mensaje de LangChain al contrato de la API."""
    return Message(
        seq=seq,
        role=_ROLES.get(message.type, "assistant"),
        # `.text` aplana los bloques de contenido (un AIMessage puede venir como
        # lista de bloques) a la cadena que el usuario efectivamente vio.
        content=message.text,
        created_at=created_at,
        tool_calls=getattr(message, "tool_calls", None) or None,
        tool_call_id=getattr(message, "tool_call_id", None),
    )


def _visible(message: BaseMessage) -> bool:
    """¿Es un mensaje de la conversación tal como la ve el usuario?

    Los ToolMessage y los AIMessage que sólo contienen llamadas a herramientas
    son maquinaria interna del turno. Se pueden pedir con `include_tools=true`,
    pero no son lo que alguien espera al leer "historial conversacional".
    """
    if isinstance(message, ToolMessage):
        return False
    if isinstance(message, AIMessage) and message.tool_calls and not message.text:
        return False
    return True


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness del servicio",
    tags=["operación"],
)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.environment,
        model=settings.model,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Enviar un mensaje a un hilo de conversación",
    tags=["chat"],
    responses={503: {"model": ErrorResponse, "description": "Modelo no disponible"}},
)
async def post_chat(
    payload: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Procesa un mensaje y devuelve la respuesta del agente.

    Si no se manda `conversation_id`, se crea un hilo nuevo y su id viene en la
    respuesta: el cliente lo reusa en los mensajes siguientes para que el
    agente recuerde el contexto.
    """
    conversation_id = payload.conversation_id or uuid.uuid4().hex
    conversation_id_var.set(conversation_id)

    turn = await service.send(conversation_id, payload.message)
    result = turn.result

    return ChatResponse(
        conversation_id=conversation_id,
        reply=result.output,
        iterations=result.iterations,
        stop_reason=result.stop_reason,
        steps=[
            ToolStep(
                iteration=step.iteration,
                tool=step.tool,
                arguments=step.args,
                ok=step.ok,
                duration_ms=step.duration_ms,
                error=step.error,
            )
            for step in result.steps
        ],
        usage=Usage(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        ),
    )


@router.get(
    "/chat/{conversation_id}",
    response_model=ConversationResponse,
    summary="Historial de un hilo de conversación",
    tags=["chat"],
    responses={404: {"model": ErrorResponse, "description": "El hilo no existe"}},
)
async def get_chat(
    conversation_id: str = Path(
        ...,
        pattern=CONVERSATION_ID_PATTERN,
        description="Id del hilo devuelto por POST /chat.",
    ),
    include_tools: bool = Query(
        default=False,
        description=(
            "Incluir los pasos internos del agente (llamadas a herramientas y "
            "sus resultados) además de los turnos de la conversación."
        ),
    ),
    service: ChatService = Depends(get_chat_service),
) -> ConversationResponse:
    conversation_id_var.set(conversation_id)
    try:
        conversation: Conversation = await service.conversation(conversation_id)
    except ConversationNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No existe la conversación {conversation_id!r}.",
        )

    records = conversation.messages
    if not include_tools:
        records = [r for r in records if _visible(r.message)]

    return ConversationResponse(
        conversation_id=conversation.id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=len(records),
        messages=[_to_public(r.seq, r.message, r.created_at) for r in records],
    )
