"""Punto de entrada de la aplicación.

Todo lo caro —conexión a la base, cliente del modelo, construcción de las
tools— se hace una vez en el arranque y se guarda en `app.state`. Construirlo
por request agregaría latencia y, en el caso de la caché de las tools, la
volvería inútil.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from .agent.llm import MissingCredentialsError, build_llm
from .agent.service import ChatService
from .api import router
from .config import Settings, get_settings
from .logging_conf import configure_logging, request_id_var
from .memory import SQLiteConversationStore
from .schemas import ErrorResponse
from .tools import YahooFinanceClient, build_tools

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    store = await SQLiteConversationStore.create(settings.sqlite_path)
    market = YahooFinanceClient(settings)
    tools = build_tools(market)
    llm = build_llm(settings)

    app.state.settings = settings
    app.state.store = store
    app.state.chat_service = ChatService(llm, tools, store, settings)

    log.info(
        "servicio arriba",
        extra={"tools": [t.name for t in tools], "environment": settings.environment},
    )
    try:
        yield
    finally:
        await store.close()
        log.info("servicio detenido")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Finance Agent API",
        version="1.0.0",
        summary=(
            "Agente conversacional con memoria por hilo y acceso a datos de "
            "mercado de Yahoo Finance."
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Correlación y traza de acceso.

        El id llega del cliente si lo manda (para poder seguir un pedido a
        través de varios servicios) y si no se genera acá. Vuelve siempre en la
        respuesta: sin eso, un usuario que reporta un error no tiene nada que
        darnos para encontrarlo en los logs.
        """
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if 0 < len(incoming) <= 64 else uuid.uuid4().hex
        request_id_var.set(request_id)

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception(
                "request falló",
                extra={"method": request.method, "route": request.url.path},
            )
            raise
        duration_ms = int((time.perf_counter() - started) * 1000)

        response.headers["x-request-id"] = request_id
        if request.url.path != "/health":
            log.info(
                "request",
                extra={
                    "method": request.method,
                    "route": request.url.path,
                    "status": response.status_code,
                    "duration_ms": duration_ms,
                },
            )
        return response

    @app.exception_handler(MissingCredentialsError)
    async def _missing_credentials(request: Request, exc: MissingCredentialsError):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                detail=str(exc),
                code="model_unavailable",
                request_id=request_id_var.get(),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        """Nada de tracebacks en la respuesta.

        El cliente recibe el request_id; el detalle está en los logs contra ese
        mismo id.
        """
        log.exception("error no controlado", extra={"route": request.url.path})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                detail="Error interno del servicio.",
                code="internal_error",
                request_id=request_id_var.get(),
            ).model_dump(),
        )

    app.include_router(router)
    return app


app = create_app()
