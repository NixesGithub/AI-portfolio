"""Utilidades comunes de los tests.

Regla: los tests no tocan la red ni necesitan credenciales. El modelo es el
``FakeChatModel`` y Yahoo Finance se sustituye por un ticker de mentira, así que
lo que se ejercita es todo el circuito real menos las dos fronteras externas.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.tools import yahoo_finance  # noqa: E402
from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402


class FakeTicker:
    """Doble de ``yfinance.Ticker`` con datos fijos."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    @property
    def fast_info(self) -> dict[str, Any]:
        return {
            "last_price": 191.25,
            "previous_close": 188.0,
            "day_high": 192.4,
            "day_low": 187.9,
            "year_high": 199.62,
            "year_low": 164.08,
            "market_cap": 2_950_000_000_000,
            "currency": "USD",
            "exchange": "NMS",
        }

    def history(self, period: str = "1mo", interval: str = "1d", **_: Any):
        index = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "Close": [100.0 + i for i in range(40)],
                "High": [101.0 + i for i in range(40)],
                "Low": [99.0 + i for i in range(40)],
                "Volume": [1_000_000 + i for i in range(40)],
            },
            index=index,
        )

    @property
    def info(self) -> dict[str, Any]:
        return {
            "longName": "Apple Inc.",
            "quoteType": "EQUITY",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "country": "United States",
            "website": "https://www.apple.com",
            "fullTimeEmployees": 161_000,
            "marketCap": 2_950_000_000_000,
            "currency": "USD",
            "longBusinessSummary": "Apple diseña, fabrica y vende dispositivos. " * 40,
        }


class EmptyTicker(FakeTicker):
    """Lo que devuelve Yahoo para un símbolo que no existe: nada."""

    @property
    def fast_info(self) -> dict[str, Any]:
        return {}

    def history(self, period: str = "1mo", interval: str = "1d", **_: Any):
        return pd.DataFrame()

    @property
    def info(self) -> dict[str, Any]:
        return {}


@pytest.fixture(autouse=True)
def clear_tool_cache():
    yahoo_finance._cache.clear()
    yield
    yahoo_finance._cache.clear()


@pytest.fixture
def fake_yahoo(monkeypatch):
    """Sustituye Yahoo Finance. ``calls`` cuenta las consultas reales (para la caché)."""
    calls: list[str] = []

    def factory(symbol: str):
        calls.append(symbol)
        return EmptyTicker(symbol) if symbol.startswith("NOPE") else FakeTicker(symbol)

    monkeypatch.setattr(yahoo_finance, "_get_ticker", factory)
    return calls


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="fake",
        store_backend="memory",
        environment="local",
        log_format="text",
        api_key="",
        agent_max_iterations=4,
        agent_timeout_s=10,
        tool_timeout_s=5,
    )


@pytest.fixture
def client(settings, fake_yahoo) -> TestClient:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
