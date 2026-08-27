"""Tests de las tools de Yahoo Finance.

yfinance está mockeado: un test que dependa del mercado real falla los sábados,
cuando Yahoo rate-limitea o cuando cambia un campo del JSON. Lo que se verifica
acá es nuestro código —el mapeo de campos, la traducción de errores, el
recorte de series—, no que Yahoo funcione.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.config import Settings
from app.tools import YahooFinanceClient, build_tools
from app.tools.yahoo_finance import MarketDataError, _finite, _round


@pytest.fixture
def client() -> YahooFinanceClient:
    return YahooFinanceClient(Settings(max_history_rows=10))


@pytest.fixture
def tools(client):
    return {t.name: t for t in build_tools(client)}


class FakeFastInfo:
    def __init__(self, data: dict | None):
        self._data = data

    def get(self, key):
        if self._data is None:
            # Así rompe yfinance cuando el símbolo no existe: no devuelve
            # vacío, revienta al indexar una respuesta incompleta.
            raise KeyError("currentTradingPeriod")
        return self._data.get(key)


def test_finite_descarta_nan_e_infinito():
    # NaN es un valor normal en pandas y JSON inválido: si se cuela, el cliente
    # recibe un cuerpo que su parser rechaza.
    assert _finite(float("nan")) is None
    assert _finite(float("inf")) is None
    assert _finite(3.5) == 3.5
    assert _finite(None) is None
    assert _finite("USD") == "USD"
    assert _round(3.14159, 2) == 3.14


async def test_get_quote_mapea_los_campos(monkeypatch, client, tools):
    monkeypatch.setattr(
        "app.tools.yahoo_finance.yf.Ticker",
        lambda symbol: type("T", (), {"fast_info": FakeFastInfo({
            "lastPrice": 102.5, "previousClose": 100.0, "open": 100.5,
            "dayHigh": 103.0, "dayLow": 99.5, "lastVolume": 1_000_000,
            "marketCap": 5_000_000_000, "currency": "USD", "exchange": "NMS",
            "quoteType": "EQUITY", "yearHigh": 120.0, "yearLow": 80.0,
            "fiftyDayAverage": 101.0, "twoHundredDayAverage": 95.0,
        })})(),
    )

    payload = json.loads(await tools["get_quote"].ainvoke({"symbol": "aapl"}))

    assert payload["symbol"] == "AAPL"  # normalizado
    assert payload["price"] == 102.5
    assert payload["change"] == 2.5
    assert payload["change_percent"] == 2.5
    assert payload["currency"] == "USD"
    assert "as_of" in payload


async def test_get_quote_con_simbolo_inexistente_sugiere_buscar(monkeypatch, tools):
    monkeypatch.setattr(
        "app.tools.yahoo_finance.yf.Ticker",
        lambda symbol: type("T", (), {"fast_info": FakeFastInfo(None)})(),
    )

    payload = json.loads(await tools["get_quote"].ainvoke({"symbol": "NOPE"}))

    assert payload["code"] == "symbol_not_found"
    # El mensaje de error es para el modelo: tiene que decirle cómo seguir.
    assert "search_symbol" in payload["error"]


async def test_los_fallos_de_red_no_se_confunden_con_simbolo_inexistente(
    monkeypatch, client
):
    def explota(symbol):
        raise ConnectionError("se cayó la red")

    monkeypatch.setattr("app.tools.yahoo_finance.yf.Ticker", explota)

    with pytest.raises(MarketDataError) as excinfo:
        await client.quote("AAPL")
    assert excinfo.value.code == "upstream_error"


async def test_la_serie_historica_se_submuestrea(monkeypatch, tools):
    """Una serie larga entera se paga por token en cada turno posterior."""
    index = pd.date_range("2024-01-01", periods=200, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "Open": range(200), "High": range(1, 201), "Low": range(200),
            "Close": [float(i) for i in range(100, 300)],
            "Volume": [1000] * 200,
        },
        index=index,
    )
    monkeypatch.setattr(
        "app.tools.yahoo_finance.yf.Ticker",
        lambda symbol: type("T", (), {"history": lambda self, **kw: frame})(),
    )

    payload = json.loads(await tools["get_price_history"].ainvoke(
        {"symbol": "AAPL", "period": "1y", "interval": "1d"}))

    assert payload["observations"] == 200
    assert payload["returned_rows"] <= 11  # max_history_rows=10, + la última
    assert len(payload["series"]) == payload["returned_rows"]
    # La última observación no se pierde nunca: es la que el usuario pregunta.
    assert payload["series"][-1]["date"].startswith("2024-07-18")
    assert payload["last_close"] == 299.0
    assert payload["note"]


async def test_serie_vacia_devuelve_error_utilizable(monkeypatch, tools):
    monkeypatch.setattr(
        "app.tools.yahoo_finance.yf.Ticker",
        lambda symbol: type("T", (), {
            "history": lambda self, **kw: pd.DataFrame()})(),
    )

    payload = json.loads(await tools["get_price_history"].ainvoke(
        {"symbol": "AAPL", "period": "1d", "interval": "1m"}))

    assert payload["code"] == "no_data"


async def test_search_symbol_normaliza_resultados(monkeypatch, tools):
    monkeypatch.setattr(
        "app.tools.yahoo_finance.yf.Search",
        lambda query, max_results: type("S", (), {"quotes": [
            {"symbol": "AAPL", "longname": "Apple Inc.", "exchDisp": "NASDAQ",
             "quoteType": "EQUITY"},
            {"shortname": "Sin símbolo"},  # se descarta
        ]})(),
    )

    payload = json.loads(await tools["search_symbol"].ainvoke({"query": "Apple"}))

    assert len(payload["results"]) == 1
    assert payload["results"][0] == {
        "symbol": "AAPL", "name": "Apple Inc.",
        "exchange": "NASDAQ", "type": "EQUITY",
    }


async def test_la_cache_evita_repreguntarle_a_yahoo(monkeypatch, client):
    llamadas = {"n": 0}

    def contar(symbol):
        llamadas["n"] += 1
        return type("T", (), {"fast_info": FakeFastInfo({
            "lastPrice": 100.0, "previousClose": 100.0})})()

    monkeypatch.setattr("app.tools.yahoo_finance.yf.Ticker", contar)

    await client.quote("AAPL")
    await client.quote("AAPL")
    await client.quote("aapl")  # misma clave, distinta capitalización

    assert llamadas["n"] == 1
