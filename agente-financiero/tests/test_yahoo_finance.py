"""Tests de la tool de Yahoo Finance (con la red sustituida)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.agent.tools.yahoo_finance import YahooFinanceInput, YahooFinanceTool


@pytest.fixture
def tool() -> YahooFinanceTool:
    return YahooFinanceTool(max_history_points=10)


def test_quote_normaliza_la_respuesta(tool, fake_yahoo):
    payload = json.loads(tool.invoke({"symbol": "AAPL", "section": "quote"}))

    assert payload["symbol"] == "AAPL"
    assert payload["last_price"] == 191.25
    assert payload["change"] == pytest.approx(3.25)
    assert payload["change_pct"] == pytest.approx(1.7287, abs=1e-3)
    assert payload["currency"] == "USD"
    assert payload["as_of"]


def test_history_resume_y_muestrea_la_serie(tool, fake_yahoo):
    payload = json.loads(
        tool.invoke({"symbol": "AAPL", "section": "history", "period": "3mo"})
    )

    assert payload["points"] == 40
    assert len(payload["series"]) == 10  # 40 velas muestreadas a 10 puntos
    assert payload["first_close"] == 100.0
    assert payload["last_close"] == 139.0
    assert payload["series"][-1]["close"] == 139.0  # el último punto no se pierde
    assert payload["change_pct"] == pytest.approx(39.0)


def test_profile_recorta_el_resumen(tool, fake_yahoo):
    payload = json.loads(tool.invoke({"symbol": "AAPL", "section": "profile"}))

    assert payload["name"] == "Apple Inc."
    assert payload["sector"] == "Technology"
    assert len(payload["summary"]) <= 801


def test_simbolo_inexistente_devuelve_error_para_el_modelo(tool, fake_yahoo):
    payload = json.loads(tool.invoke({"symbol": "NOPE", "section": "quote"}))

    assert payload["error"] == "symbol_not_found"
    assert "hint" in payload  # el modelo necesita saber cómo corregirlo


def test_un_fallo_de_yahoo_no_lanza_excepcion(tool, monkeypatch):
    from app.agent.tools import yahoo_finance

    def explode(symbol: str):
        raise ConnectionError("Yahoo caído")

    monkeypatch.setattr(yahoo_finance, "_get_ticker", explode)
    payload = json.loads(tool.invoke({"symbol": "AAPL"}))

    assert payload["error"] == "upstream_error"
    assert "no inventes" in payload["hint"]


def test_la_cache_evita_llamadas_repetidas(tool, fake_yahoo):
    tool.invoke({"symbol": "AAPL", "section": "quote"})
    tool.invoke({"symbol": "AAPL", "section": "quote"})
    tool.invoke({"symbol": "MSFT", "section": "quote"})

    assert fake_yahoo == ["AAPL", "MSFT"]


def test_el_simbolo_se_normaliza_y_se_valida():
    assert YahooFinanceInput(symbol=" aapl ").symbol == "AAPL"
    assert YahooFinanceInput(symbol="BTC-USD").symbol == "BTC-USD"
    assert YahooFinanceInput(symbol="^GSPC").symbol == "^GSPC"

    for malo in ("", "DROP TABLE users", "AAPL; rm -rf /", "x" * 20):
        with pytest.raises(ValidationError):
            YahooFinanceInput(symbol=malo)


def test_la_descripcion_orienta_al_modelo(tool):
    schema = tool.args_schema.model_json_schema()
    assert "yahoo_finance" == tool.name
    assert "quote" in schema["properties"]["section"]["description"]
    assert "BTC-USD" in schema["properties"]["symbol"]["description"]


async def test_version_asincrona(tool, fake_yahoo):
    payload = json.loads(await tool.ainvoke({"symbol": "AAPL"}))
    assert payload["last_price"] == 191.25
