"""Registro de tools del agente.

Añadir una capacidad nueva (noticias, conversión de divisas, una API interna) es
escribir la tool y registrarla aquí: ni el bucle del agente ni la API cambian.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.agent.tools.yahoo_finance import YahooFinanceTool, build_yahoo_finance_tool

__all__ = ["YahooFinanceTool", "build_tools", "build_yahoo_finance_tool"]


def build_tools(settings) -> list[BaseTool]:
    return [build_yahoo_finance_tool(settings)]
