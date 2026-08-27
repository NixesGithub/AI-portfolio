"""Tool de consulta a Yahoo Finance.

Una sola tool con tres secciones (``quote``, ``history``, ``profile``) en lugar de
tres tools: al modelo le resulta más fácil elegir bien entre pocas herramientas
bien descritas, y el esquema tipado (``section``, ``period``, ``interval``) evita
la mayoría de las llamadas inválidas.

Decisiones que importan en producción:

* **La salida se normaliza**. El modelo recibe un JSON pequeño y estable, no el
  volcado de yfinance: menos tokens y menos alucinación sobre campos raros.
* **Las series se recortan**. Un año de velas diarias son 250 filas; se muestrean
  a ``yf_max_history_points`` puntos conservando máximo, mínimo y último.
* **Los errores son texto para el modelo, no excepciones**. Si el símbolo no
  existe, el agente debe poder pedir una aclaración en vez de romper la petición.
* **yfinance es bloqueante**, así que la versión async lo manda a un hilo.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, Literal

import yfinance as yf
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

from app.agent.tools.cache import TTLCache

logger = logging.getLogger(__name__)

SYMBOL_RE = re.compile(r"^[A-Za-z0-9.\-=^]{1,16}$")

PERIODS = ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
INTERVALS = ("5m", "15m", "1h", "1d", "1wk", "1mo")

_cache = TTLCache()


class YahooFinanceError(Exception):
    """Fallo esperable de la tool; se le devuelve al modelo como texto."""


class SymbolNotFound(YahooFinanceError):
    pass


class YahooFinanceInput(BaseModel):
    """Argumentos de la tool (esto es lo que ve el modelo)."""

    symbol: str = Field(
        description=(
            "Ticker de Yahoo Finance. Acciones: AAPL, MSFT, TSLA. Índices: ^GSPC, "
            "^IBEX. Divisas: EURUSD=X. Cripto: BTC-USD. Bolsas fuera de EE. UU. "
            "llevan sufijo: SAN.MC, VOD.L."
        )
    )
    section: Literal["quote", "history", "profile"] = Field(
        default="quote",
        description=(
            "quote: precio actual, variación del día y rango de 52 semanas. "
            "history: evolución del precio en un periodo. "
            "profile: qué es la empresa (sector, industria, país, tamaño)."
        ),
    )
    period: Literal[PERIODS] = Field(  # type: ignore[valid-type]
        default="1mo", description="Sólo para section=history. Ventana temporal."
    )
    interval: Literal[INTERVALS] = Field(  # type: ignore[valid-type]
        default="1d",
        description=(
            "Sólo para section=history. Granularidad de las velas. Los intervalos "
            "intradía (5m, 15m, 1h) sólo existen para periodos cortos."
        ),
    )

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        value = value.strip().upper()
        if not SYMBOL_RE.match(value):
            raise ValueError(
                "Símbolo inválido: usa un ticker de Yahoo Finance como AAPL o BTC-USD."
            )
        return value


def _get_ticker(symbol: str) -> Any:
    """Punto de indirección para poder sustituir yfinance en los tests."""
    return yf.Ticker(symbol)


def _num(value: Any) -> float | None:
    """Normaliza a float; descarta NaN e infinitos (yfinance devuelve ambos)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return round(number, 6)


def _pct_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0):
        return None
    return round((current - previous) / previous * 100, 4)


def _fast(fast_info: Any, key: str) -> Any:
    try:
        return fast_info[key]
    except Exception:  # noqa: BLE001 - FastInfo lanza de todo cuando falta el dato
        return None


def _quote(symbol: str) -> dict[str, Any]:
    ticker = _get_ticker(symbol)
    fast_info = ticker.fast_info
    last = _num(_fast(fast_info, "last_price"))
    previous_close = _num(_fast(fast_info, "previous_close"))
    if last is None and previous_close is None:
        raise SymbolNotFound(
            f"Yahoo Finance no devuelve cotización para '{symbol}'."
        )
    return {
        "symbol": symbol,
        "section": "quote",
        "currency": _fast(fast_info, "currency"),
        "exchange": _fast(fast_info, "exchange"),
        "last_price": last,
        "previous_close": previous_close,
        "change": None if last is None or previous_close is None
        else round(last - previous_close, 6),
        "change_pct": _pct_change(last, previous_close),
        "day_high": _num(_fast(fast_info, "day_high")),
        "day_low": _num(_fast(fast_info, "day_low")),
        "year_high": _num(_fast(fast_info, "year_high")),
        "year_low": _num(_fast(fast_info, "year_low")),
        "market_cap": _num(_fast(fast_info, "market_cap")),
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Yahoo Finance",
    }


def _downsample(rows: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if len(rows) <= max_points:
        return rows
    step = len(rows) / max_points
    sampled = [rows[min(int(i * step), len(rows) - 1)] for i in range(max_points)]
    if sampled[-1] is not rows[-1]:
        sampled[-1] = rows[-1]
    return sampled


def _history(symbol: str, period: str, interval: str, max_points: int) -> dict[str, Any]:
    ticker = _get_ticker(symbol)
    frame = ticker.history(period=period, interval=interval, auto_adjust=True)
    if frame is None or len(frame) == 0:
        raise SymbolNotFound(
            f"Yahoo Finance no devuelve histórico para '{symbol}' "
            f"(period={period}, interval={interval})."
        )
    rows: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        close = _num(row.get("Close"))
        if close is None:
            continue
        rows.append(
            {
                "date": index.isoformat() if hasattr(index, "isoformat") else str(index),
                "close": close,
                "high": _num(row.get("High")),
                "low": _num(row.get("Low")),
                "volume": _num(row.get("Volume")),
            }
        )
    if not rows:
        raise SymbolNotFound(f"El histórico de '{symbol}' llegó vacío.")

    closes = [row["close"] for row in rows]
    first, last = closes[0], closes[-1]
    return {
        "symbol": symbol,
        "section": "history",
        "period": period,
        "interval": interval,
        "points": len(rows),
        "start": rows[0]["date"],
        "end": rows[-1]["date"],
        "first_close": first,
        "last_close": last,
        "change_pct": _pct_change(last, first),
        "max_close": max(closes),
        "min_close": min(closes),
        "series": _downsample(rows, max_points),
        "series_note": (
            f"Serie muestreada a {max_points} puntos sobre {len(rows)} velas."
            if len(rows) > max_points
            else "Serie completa."
        ),
        "source": "Yahoo Finance",
    }


def _profile(symbol: str) -> dict[str, Any]:
    ticker = _get_ticker(symbol)
    info: dict[str, Any] = ticker.info or {}
    if not info.get("longName") and not info.get("shortName"):
        raise SymbolNotFound(f"Yahoo Finance no tiene ficha para '{symbol}'.")
    summary = info.get("longBusinessSummary") or ""
    return {
        "symbol": symbol,
        "section": "profile",
        "name": info.get("longName") or info.get("shortName"),
        "quote_type": info.get("quoteType"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "website": info.get("website"),
        "employees": info.get("fullTimeEmployees"),
        "market_cap": _num(info.get("marketCap")),
        "currency": info.get("currency"),
        "summary": summary[:800] + ("…" if len(summary) > 800 else ""),
        "source": "Yahoo Finance",
    }


class YahooFinanceTool(BaseTool):
    """Datos de mercado de Yahoo Finance."""

    name: str = "yahoo_finance"
    description: str = (
        "Consulta datos de mercado reales en Yahoo Finance para un ticker: "
        "cotización actual (quote), evolución histórica (history) o ficha de la "
        "empresa (profile). Úsala siempre que haga falta un precio, una variación, "
        "una capitalización o cualquier dato de mercado: nunca los inventes ni los "
        "cites de memoria. Devuelve JSON."
    )
    args_schema: type[BaseModel] = YahooFinanceInput

    ttl_quote: int = 60
    ttl_history: int = 300
    ttl_profile: int = 86_400
    max_history_points: int = 30

    def _run(  # type: ignore[override]
        self,
        symbol: str,
        section: str = "quote",
        period: str = "1mo",
        interval: str = "1d",
        **_: Any,
    ) -> str:
        key = f"{section}:{symbol}:{period}:{interval}"
        cached = _cache.get(key)
        if cached is not None:
            logger.debug("yahoo_finance cache hit", extra={"tool_key": key})
            return cached

        try:
            if section == "quote":
                payload, ttl = _quote(symbol), self.ttl_quote
            elif section == "history":
                payload = _history(symbol, period, interval, self.max_history_points)
                ttl = self.ttl_history
            elif section == "profile":
                payload, ttl = _profile(symbol), self.ttl_profile
            else:  # pragma: no cover - Literal lo impide antes de llegar aquí
                raise YahooFinanceError(f"Sección desconocida: {section}")
        except SymbolNotFound as exc:
            return json.dumps(
                {
                    "error": "symbol_not_found",
                    "message": str(exc),
                    "hint": (
                        "Puede ser un ticker mal escrito o que Yahoo no lo cubra. "
                        "Las bolsas fuera de EE. UU. llevan sufijo (SAN.MC, VOD.L) "
                        "y las cripto van como BTC-USD. Confírmalo con el usuario "
                        "en vez de dar por buenos unos datos que no tienes."
                    ),
                },
                ensure_ascii=False,
            )
        except Exception as exc:  # noqa: BLE001 - yfinance lanza excepciones sin tipar
            logger.warning(
                "yahoo_finance falló", extra={"tool_key": key, "error": str(exc)}
            )
            return json.dumps(
                {
                    "error": "upstream_error",
                    "message": (
                        "Yahoo Finance no está respondiendo ahora mismo: "
                        f"{type(exc).__name__}."
                    ),
                    "hint": "Díselo al usuario; no inventes los datos.",
                },
                ensure_ascii=False,
            )

        result = json.dumps(payload, ensure_ascii=False)
        _cache.set(key, result, ttl)
        return result

    async def _arun(  # type: ignore[override]
        self,
        symbol: str,
        section: str = "quote",
        period: str = "1mo",
        interval: str = "1d",
        **kwargs: Any,
    ) -> str:
        # yfinance es síncrono y hace I/O de red: fuera del event loop.
        return await asyncio.to_thread(
            self._run, symbol, section, period, interval, **kwargs
        )


def build_yahoo_finance_tool(settings) -> YahooFinanceTool:
    return YahooFinanceTool(
        ttl_quote=settings.yf_cache_ttl_quote_s,
        ttl_history=settings.yf_cache_ttl_history_s,
        ttl_profile=settings.yf_cache_ttl_profile_s,
        max_history_points=settings.yf_max_history_points,
    )
