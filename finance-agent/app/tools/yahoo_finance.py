"""Acceso a datos de mercado de Yahoo Finance, expuesto como tools de LangChain.

Tres decisiones que valen la pena explicar:

1. **yfinance es síncrono y hace red.** Llamarlo directo desde una corrutina
   congela el event loop y con él a todos los demás pedidos del proceso. Cada
   llamada va a un thread (`asyncio.to_thread`) con timeout.

2. **Las tools nunca propagan una excepción al bucle del agente.** Devuelven
   `{"error": ..., "code": ...}`. Un símbolo mal escrito no es un fallo del
   servicio: es información que el modelo puede usar para corregirse solo
   (buscar el ticker y reintentar). Los fallos de infraestructura sí se
   loguean con nivel de error.

3. **Las respuestas vienen recortadas.** Un `history` de un año son 250 filas
   que el modelo no necesita y que se pagan por token en cada turno siguiente,
   porque quedan en el historial. Se manda un resumen estadístico y una
   muestra acotada.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Literal

import yfinance as yf
from langchain_core.tools import BaseTool, tool

from ..config import Settings
from .cache import TTLCache

log = logging.getLogger(__name__)

Period = Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"]
Interval = Literal["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"]


class MarketDataError(Exception):
    """Fallo al consultar Yahoo. `code` distingue 'no existe' de 'se cayó'."""

    def __init__(self, message: str, code: str = "market_data_error") -> None:
        super().__init__(message)
        self.code = code


def _finite(value: Any) -> Any:
    """Convierte a float JSON-serializable, o None.

    NaN e infinito son valores normales en pandas y JSON inválido en la RFC:
    `json.dumps` los escribe como `NaN`, que después revienta del otro lado.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if math.isfinite(number) else None


def _round(value: Any, digits: int = 4) -> Any:
    number = _finite(value)
    return round(number, digits) if isinstance(number, float) else number


class YahooFinanceClient:
    """Cliente fino sobre yfinance: threads, timeouts y caché."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._quotes = TTLCache(settings.quote_cache_ttl_seconds)
        self._reference = TTLCache(settings.reference_cache_ttl_seconds)

    async def _call(self, fn, *args: Any) -> Any:
        """Corre una función bloqueante de yfinance con timeout.

        Todo lo que yfinance pueda levantar termina como MarketDataError: es
        una librería que scrapea una API no documentada y rompe de maneras
        creativas (KeyError sobre un dict incompleto, IndexError sobre un
        DataFrame vacío). El bucle del agente no tiene que conocer ninguna de
        esas formas.
        """
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(fn, *args),
                timeout=self._settings.tool_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise MarketDataError(
                f"Yahoo Finance no respondió en {self._settings.tool_timeout_seconds:.0f}s",
                code="upstream_timeout",
            ) from exc
        except MarketDataError:
            raise
        except Exception as exc:
            # Esto sí es un fallo nuestro o de Yahoo, no del usuario: se loguea.
            log.error(
                "fallo consultando Yahoo Finance",
                extra={"error_type": type(exc).__name__},
                exc_info=True,
            )
            raise MarketDataError(
                "Yahoo Finance devolvió una respuesta inesperada. Reintentá en unos "
                "segundos o probá con otro símbolo.",
                code="upstream_error",
            ) from exc

    # -- operaciones ---------------------------------------------------------

    async def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        def _fetch() -> list[dict[str, Any]]:
            return list(yf.Search(query, max_results=limit).quotes or [])

        raw = await self._reference.get_or_set(
            ("search", query.lower(), limit), lambda: self._call(_fetch)
        )
        return [
            {
                "symbol": item.get("symbol"),
                "name": item.get("longname") or item.get("shortname"),
                "exchange": item.get("exchDisp"),
                "type": item.get("quoteType"),
            }
            for item in raw
            if item.get("symbol")
        ]

    async def quote(self, symbol: str) -> dict[str, Any]:
        def _fetch() -> dict[str, Any]:
            info = yf.Ticker(symbol).fast_info
            fields = (
                "lastPrice", "previousClose", "open", "dayHigh", "dayLow",
                "lastVolume", "marketCap", "currency", "exchange", "quoteType",
                "yearHigh", "yearLow", "fiftyDayAverage", "twoHundredDayAverage",
            )
            try:
                # Para un símbolo que no existe, yfinance no devuelve vacío:
                # revienta al indexar la respuesta incompleta de Yahoo. Esa
                # excepción es el chequeo de existencia.
                snapshot = {key: info.get(key) for key in fields}
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise MarketDataError(
                    f"No hay datos de mercado para el símbolo {symbol!r}. "
                    "Puede no existir o estar deslistado: probá search_symbol "
                    "para encontrar el ticker correcto.",
                    code="symbol_not_found",
                ) from exc

            if snapshot.get("lastPrice") is None:
                raise MarketDataError(
                    f"No hay datos de mercado para el símbolo {symbol!r}. "
                    "Puede no existir o estar deslistado: probá search_symbol "
                    "para encontrar el ticker correcto.",
                    code="symbol_not_found",
                )
            return snapshot

        raw = await self._quotes.get_or_set(
            ("quote", symbol.upper()), lambda: self._call(_fetch)
        )

        last = _finite(raw.get("lastPrice"))
        previous = _finite(raw.get("previousClose"))
        change = last - previous if last is not None and previous else None

        return {
            "symbol": symbol.upper(),
            "price": _round(last, 4),
            "previous_close": _round(previous, 4),
            "change": _round(change, 4),
            "change_percent": _round(change / previous * 100, 2)
            if change is not None and previous
            else None,
            "open": _round(raw.get("open"), 4),
            "day_high": _round(raw.get("dayHigh"), 4),
            "day_low": _round(raw.get("dayLow"), 4),
            "volume": _finite(raw.get("lastVolume")),
            "market_cap": _finite(raw.get("marketCap")),
            "fifty_two_week_high": _round(raw.get("yearHigh"), 4),
            "fifty_two_week_low": _round(raw.get("yearLow"), 4),
            "fifty_day_average": _round(raw.get("fiftyDayAverage"), 4),
            "two_hundred_day_average": _round(raw.get("twoHundredDayAverage"), 4),
            "currency": raw.get("currency"),
            "exchange": raw.get("exchange"),
            "quote_type": raw.get("quoteType"),
            "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    async def history(
        self, symbol: str, period: str, interval: str
    ) -> dict[str, Any]:
        max_rows = self._settings.max_history_rows

        def _fetch() -> dict[str, Any]:
            frame = yf.Ticker(symbol).history(period=period, interval=interval)
            if frame.empty:
                raise MarketDataError(
                    f"Yahoo no devolvió series para {symbol!r} con period={period} "
                    f"interval={interval}. La combinación puede ser inválida "
                    "(los intervalos intradiarios sólo cubren los últimos días).",
                    code="no_data",
                )

            closes = frame["Close"].dropna()
            first_close, last_close = float(closes.iloc[0]), float(closes.iloc[-1])

            # Downsampling: filas repartidas uniformemente, nunca más de
            # `max_rows`, y siempre incluyendo la última (que es la que el
            # usuario tiene en la cabeza cuando pregunta "cómo cerró").
            step = math.ceil(len(frame) / max_rows) if max_rows else 1
            positions = list(range(0, len(frame), max(step, 1)))
            if positions[-1] != len(frame) - 1:
                positions.append(len(frame) - 1)
            sampled = frame.iloc[positions]

            return {
                "rows": [
                    {
                        "date": index.isoformat(),
                        "open": _round(row.Open, 4),
                        "high": _round(row.High, 4),
                        "low": _round(row.Low, 4),
                        "close": _round(row.Close, 4),
                        "volume": _finite(row.Volume),
                    }
                    for index, row in zip(sampled.index, sampled.itertuples())
                ],
                "first_close": first_close,
                "last_close": last_close,
                "high": float(frame["High"].max()),
                "low": float(frame["Low"].min()),
                "observations": int(len(frame)),
                "start": frame.index[0].isoformat(),
                "end": frame.index[-1].isoformat(),
            }

        data = await self._quotes.get_or_set(
            ("history", symbol.upper(), period, interval), lambda: self._call(_fetch)
        )

        first, last = data["first_close"], data["last_close"]
        return {
            "symbol": symbol.upper(),
            "period": period,
            "interval": interval,
            "start": data["start"],
            "end": data["end"],
            "observations": data["observations"],
            "returned_rows": len(data["rows"]),
            "first_close": _round(first, 4),
            "last_close": _round(last, 4),
            "change_percent": _round((last - first) / first * 100, 2) if first else None,
            "period_high": _round(data["high"], 4),
            "period_low": _round(data["low"], 4),
            "series": data["rows"],
            "note": (
                f"Serie submuestreada: {len(data['rows'])} de {data['observations']} "
                "observaciones."
            )
            if len(data["rows"]) < data["observations"]
            else None,
        }

    async def fundamentals(self, symbol: str) -> dict[str, Any]:
        def _fetch() -> dict[str, Any]:
            try:
                info = yf.Ticker(symbol).info
            except Exception as exc:  # incluye el 404 de quoteSummary
                raise MarketDataError(
                    f"No hay fundamentales para {symbol!r}. Puede no existir, o "
                    "ser un ETF, índice o cripto: esos no publican estos datos.",
                    code="symbol_not_found",
                ) from exc
            if not info or not info.get("symbol"):
                raise MarketDataError(
                    f"No hay fundamentales para {symbol!r}. Puede no existir, o "
                    "ser un ETF, índice o cripto: esos no publican estos datos.",
                    code="symbol_not_found",
                )
            return info

        info = await self._reference.get_or_set(
            ("fundamentals", symbol.upper()), lambda: self._call(_fetch)
        )

        summary = info.get("longBusinessSummary") or ""
        return {
            "symbol": symbol.upper(),
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "currency": info.get("currency"),
            "market_cap": _finite(info.get("marketCap")),
            "enterprise_value": _finite(info.get("enterpriseValue")),
            "trailing_pe": _round(info.get("trailingPE"), 2),
            "forward_pe": _round(info.get("forwardPE"), 2),
            "price_to_book": _round(info.get("priceToBook"), 2),
            "trailing_eps": _round(info.get("trailingEps"), 2),
            "dividend_yield": _round(info.get("dividendYield"), 4),
            "profit_margin": _round(info.get("profitMargins"), 4),
            "operating_margin": _round(info.get("operatingMargins"), 4),
            "return_on_equity": _round(info.get("returnOnEquity"), 4),
            "revenue": _finite(info.get("totalRevenue")),
            "revenue_growth": _round(info.get("revenueGrowth"), 4),
            "debt_to_equity": _round(info.get("debtToEquity"), 2),
            "beta": _round(info.get("beta"), 3),
            "analyst_recommendation": info.get("recommendationKey"),
            "analyst_target_mean": _round(info.get("targetMeanPrice"), 2),
            "employees": info.get("fullTimeEmployees"),
            "website": info.get("website"),
            "business_summary": summary[:600] + "…" if len(summary) > 600 else summary,
        }


def _as_json(payload: dict[str, Any] | list[Any]) -> str:
    """Serializa la salida de una tool, sin claves en None para no gastar tokens."""
    if isinstance(payload, dict):
        payload = {k: v for k, v in payload.items() if v is not None}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def _failure(exc: MarketDataError) -> str:
    return _as_json({"error": str(exc), "code": exc.code})


def build_tools(client: YahooFinanceClient) -> list[BaseTool]:
    """Construye las tools atadas a un cliente.

    Son funciones internas y no tools declaradas a nivel de módulo a propósito:
    así el cliente (y su caché, y su configuración) es inyectable, y los tests
    pueden pasar uno falso sin parchear imports.
    """

    @tool
    async def search_symbol(query: str, limit: int = 5) -> str:
        """Busca el ticker de Yahoo Finance de una empresa, índice, ETF o cripto.

        Usala siempre que el usuario nombre una empresa en lugar de un ticker
        ("Apple", "el banco Santander"), o cuando otra tool devuelva
        symbol_not_found.

        Args:
            query: Nombre a buscar, por ejemplo "Apple" o "Banco Santander".
            limit: Cuántos resultados devolver (1-10).
        """
        try:
            results = await client.search(query, max(1, min(limit, 10)))
        except MarketDataError as exc:
            return _failure(exc)
        if not results:
            return _as_json({"query": query, "results": [], "note": "Sin coincidencias."})
        return _as_json({"query": query, "results": results})

    @tool
    async def get_quote(symbol: str) -> str:
        """Devuelve la cotización actual de un símbolo de Yahoo Finance.

        Incluye precio, variación contra el cierre anterior, rango del día,
        volumen, capitalización y máximos/mínimos de 52 semanas. Es la tool
        para "a cuánto cotiza X" o "cómo viene X hoy".

        Args:
            symbol: Ticker de Yahoo, por ejemplo "AAPL", "MSFT" o "BTC-USD".
        """
        try:
            return _as_json(await client.quote(symbol))
        except MarketDataError as exc:
            return _failure(exc)

    @tool
    async def get_price_history(
        symbol: str, period: Period = "1mo", interval: Interval = "1d"
    ) -> str:
        """Devuelve la evolución histórica del precio de un símbolo.

        Para preguntas sobre rendimiento, tendencia o comparaciones en el
        tiempo ("cuánto subió en el año", "cómo viene el último mes"). La serie
        viene submuestreada si el período es largo.

        Args:
            symbol: Ticker de Yahoo, por ejemplo "AAPL".
            period: Ventana temporal: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd o max.
            interval: Granularidad. Los intervalos intradiarios (1m, 5m, 15m, 1h)
                sólo están disponibles para períodos cortos y recientes.
        """
        try:
            return _as_json(await client.history(symbol, period, interval))
        except MarketDataError as exc:
            return _failure(exc)

    @tool
    async def get_fundamentals(symbol: str) -> str:
        """Devuelve los fundamentales de una empresa cotizante.

        Valuación (PER, price/book), rentabilidad (márgenes, ROE), ingresos,
        deuda, beta y consenso de analistas. Para "está cara", "cuánto gana" o
        "a qué se dedica". Sólo aplica a acciones: los ETFs, índices y
        criptomonedas no publican estos datos.

        Args:
            symbol: Ticker de Yahoo de una acción, por ejemplo "MSFT".
        """
        try:
            return _as_json(await client.fundamentals(symbol))
        except MarketDataError as exc:
            return _failure(exc)

    return [search_symbol, get_quote, get_price_history, get_fundamentals]
