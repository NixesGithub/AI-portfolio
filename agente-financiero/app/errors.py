"""Errores de dominio y su traducción a respuestas HTTP."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging import request_id_var


class AppError(Exception):
    """Error controlado: se le puede enseñar al cliente tal cual."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConversationNotFound(AppError):
    status_code = 404
    code = "conversation_not_found"


class Unauthorized(AppError):
    status_code = 401
    code = "unauthorized"


class AgentTimeout(AppError):
    status_code = 504
    code = "agent_timeout"


class UpstreamError(AppError):
    """Fallo del proveedor del modelo (rate limit, corte, 5xx…)."""

    status_code = 502
    code = "upstream_error"


def _body(code: str, message: str, details: dict | None = None) -> dict:
    error: dict = {"code": code, "message": message, "request_id": request_id_var.get()}
    if details:
        error["details"] = details
    return {"error": error}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_body("validation_error", "La petición no es válida.",
                          {"errors": exc.errors()}),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_body("http_error", str(exc.detail)),
        )
