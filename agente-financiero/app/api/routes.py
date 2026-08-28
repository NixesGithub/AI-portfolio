"""Endpoints HTTP."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query, Response, status

from app.agent.service import ChatService
from app.api.deps import get_chat_service, get_settings, require_api_key
from app.api.schemas import (
    CONVERSATION_ID_PATTERN,
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    ErrorResponse,
    HealthResponse,
    MessageModel,
)
from app.config import Settings
from app.logging import conversation_id_var

router = APIRouter()

ERRORS: dict[int | str, dict] = {
    401: {"model": ErrorResponse, "description": "Falta la API key o no es válida"},
    404: {"model": ErrorResponse, "description": "La conversación no existe"},
    422: {"model": ErrorResponse, "description": "Petición inválida"},
    502: {"model": ErrorResponse, "description": "El proveedor del modelo falló"},
    504: {"model": ErrorResponse, "description": "El agente agotó su tiempo"},
}


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses=ERRORS,
    summary="Enviar un mensaje a un hilo de conversación",
    dependencies=[Depends(require_api_key)],
)
async def post_chat(
    payload: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Envía un mensaje y devuelve la respuesta del agente.

    Sin `conversation_id` se abre un hilo nuevo y su id viene en la respuesta;
    con él, el agente recuerda todo lo hablado antes en ese hilo.
    """
    if payload.conversation_id:
        conversation_id_var.set(payload.conversation_id)
    turn = await service.send_message(
        message=payload.message,
        conversation_id=payload.conversation_id,
        metadata=payload.metadata,
    )
    conversation_id_var.set(turn.conversation_id)
    return ChatResponse.from_turn(turn)


@router.get(
    "/chat/{conversation_id}",
    response_model=ConversationResponse,
    responses=ERRORS,
    summary="Historial de una conversación",
    dependencies=[Depends(require_api_key)],
)
async def get_chat(
    conversation_id: str = Path(pattern=CONVERSATION_ID_PATTERN),
    include_tools: bool = Query(
        default=False,
        description=(
            "Incluye las llamadas a herramientas y sus resultados. Por defecto se "
            "devuelve sólo la conversación entre usuario y agente."
        ),
    ),
    limit: int | None = Query(
        default=None, ge=1, le=500, description="Devuelve sólo los N más recientes."
    ),
    service: ChatService = Depends(get_chat_service),
) -> ConversationResponse:
    conversation_id_var.set(conversation_id)
    meta, stored = await service.get_conversation(conversation_id, limit=limit)
    messages = [MessageModel.from_stored(item) for item in stored]
    if not include_tools:
        # La traza de tools es ruido para un cliente de chat: se pide aparte.
        messages = [
            message
            for message in messages
            if message.role != "tool" and (message.content or message.role == "user")
        ]
    return ConversationResponse.build(meta, messages)


@router.delete(
    "/chat/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=ERRORS,
    summary="Borrar una conversación",
    dependencies=[Depends(require_api_key)],
)
async def delete_chat(
    conversation_id: str = Path(pattern=CONVERSATION_ID_PATTERN),
    service: ChatService = Depends(get_chat_service),
) -> Response:
    """Borra el hilo y su historial (retención de datos / derecho al olvido)."""
    conversation_id_var.set(conversation_id)
    await service.delete_conversation(conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/health/live", response_model=HealthResponse, summary="Liveness")
async def health_live(
    settings: Settings = Depends(get_settings),
    service: ChatService = Depends(get_chat_service),
) -> HealthResponse:
    """¿Está el proceso vivo? No toca dependencias: si esto falla, hay que reiniciar."""
    return HealthResponse(
        status="ok",
        environment=settings.environment,
        llm_provider=settings.llm_provider,
        store_backend=settings.store_backend,
        tools=service.tool_names,
    )


@router.get("/health/ready", response_model=HealthResponse, summary="Readiness")
async def health_ready(
    settings: Settings = Depends(get_settings),
    service: ChatService = Depends(get_chat_service),
) -> HealthResponse:
    """¿Puede atender tráfico? Comprueba el almacén antes de decir que sí."""
    healthy = True
    try:
        await service.store.health()
    except Exception:  # noqa: BLE001 - la salud nunca lanza, informa
        healthy = False
    return HealthResponse(
        status="ok" if healthy else "degraded",
        environment=settings.environment,
        llm_provider=settings.llm_provider,
        store_backend=settings.store_backend,
        tools=service.tool_names,
    )
