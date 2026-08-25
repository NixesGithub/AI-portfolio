#!/usr/bin/env python3
"""Resumen diario de noticias de IA: RSS -> deduplicar/rankear -> Claude -> Telegram.

Port del workflow de n8n `noticias-ia` (repo NixesGithub/notic-ia) a un script
autónomo, para poder ejecutarlo desde GitHub Actions en vez de depender de una
instancia local de n8n (y por tanto de que el portátil esté encendido).

Variables de entorno:
  ANTHROPIC_API_KEY    obligatoria salvo en --dry-run
  TELEGRAM_BOT_TOKEN   obligatoria salvo en --dry-run
  TELEGRAM_CHAT_ID     obligatoria salvo en --dry-run
  DIGEST_TZ            zona horaria del digest (por defecto Europe/Madrid)
  DIGEST_HOUR          hora local a la que debe salir (por defecto 9)
  DIGEST_MODEL         modelo de Anthropic (por defecto claude-sonnet-5)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import feedparser

ZONA = ZoneInfo(os.environ.get("DIGEST_TZ", "Europe/Madrid"))
HORA_DIGEST = int(os.environ.get("DIGEST_HOUR", "9"))
MODELO = os.environ.get("DIGEST_MODEL", "claude-sonnet-5")

# Muchos feeds rechazan el User-Agent por defecto de urllib con un 403.
AGENTE = "Mozilla/5.0 (compatible; notic-ia/1.0; +https://github.com/NixesGithub/AI-portfolio)"

LIMITE_TELEGRAM = 3800  # Telegram corta a 4096; dejamos margen.
MAX_CANDIDATOS = 40

MESES_ES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]

# "peso" = fiabilidad/relevancia editorial de la fuente (se usa para rankear).
PESOS = {
    "techcrunch.com": 3,
    "theverge.com": 3,
    "arstechnica.com": 3,
    "technologyreview.com": 3,
    "wired.com": 2,
    "venturebeat.com": 2,
    "news.ycombinator.com": 2,
    "news.google.com": 1,
}

NOMBRES = {
    "techcrunch.com": "TechCrunch",
    "theverge.com": "The Verge",
    "arstechnica.com": "Ars Technica",
    "technologyreview.com": "MIT Tech Review",
    "wired.com": "Wired",
    "venturebeat.com": "VentureBeat",
    "news.ycombinator.com": "Hacker News",
}

# Las búsquedas de "IA" en Google News se llenan de contenido bursátil sindicado
# ("¿Deberías comprar acciones de Nvidia?"). No es noticia de IA: lo penalizamos.
RUIDO = [
    "motley fool", "benzinga", "zacks", "investing.com", "seeking alpha",
    "simply wall st", "24/7 wall st", "insider monkey", "barchart",
    "the globe and mail", "nasdaq.com", "fool.com",
]

# Términos que suelen marcar una noticia relevante de IA.
CLAVES = [
    "openai", "anthropic", "claude", "chatgpt", "gpt", "gemini", "deepmind",
    "llama", "mistral", "nvidia", "microsoft", "meta", "apple", "amazon",
    "model", "modelo", "agent", "agente", "launch", "lanza", "release",
    "funding", "raise", "ronda", "valuation", "acquisition", "adquiere",
    "regulation", "regula", "ley", "lawsuit", "demanda", "benchmark",
    "open source", "código abierto", "chip", "data center", "centro de datos",
]


def log(mensaje: str) -> None:
    print(mensaje, flush=True)


def normalizar(texto: str | None) -> str:
    """Quita acentos y puntuación para poder comparar títulos entre fuentes."""
    base = unicodedata.normalize("NFD", texto or "").lower()
    base = "".join(c for c in base if unicodedata.category(c) != "Mn")
    base = re.sub(r"[^a-z0-9 ]", " ", base)
    return re.sub(r"\s+", " ", base).strip()


def limpiar_html(texto: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", "", texto or "")).strip()


def escapar(texto) -> str:
    """Telegram en modo HTML sólo permite unas pocas etiquetas: escapamos el resto."""
    return (
        str(texto if texto is not None else "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def fecha_en_espanol(dia: datetime) -> str:
    return f"{dia.day} de {MESES_ES[dia.month - 1]} de {dia.year}"


# --------------------------------------------------------------------------
# 1. Fuentes
# --------------------------------------------------------------------------

def url_google_news(consulta: str, idioma: str, pais: str, desde: str, hasta: str) -> str:
    q = urllib.parse.quote(f"{consulta} after:{desde} before:{hasta}")
    return (
        f"https://news.google.com/rss/search?q={q}"
        f"&hl={idioma}&gl={pais}&ceid={pais}:{idioma.split('-')[0]}"
    )


def construir_fuentes(referencia: datetime) -> list[dict]:
    """Genera las 9 URLs de RSS.

    Google News reordena por actualidad: sin acotar fechas devuelve casi sólo el
    día en curso. Con after:/before: alrededor del día de referencia sí lo cubre.
    """
    desde = (referencia - timedelta(days=1)).date().isoformat()
    hasta = (referencia + timedelta(days=1)).date().isoformat()

    return [
        {"fuente": "TechCrunch", "peso": 3, "url": "https://techcrunch.com/category/artificial-intelligence/feed/"},
        {"fuente": "The Verge", "peso": 3, "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"},
        {"fuente": "Ars Technica", "peso": 3, "url": "https://arstechnica.com/ai/feed/"},
        {"fuente": "MIT Tech Review", "peso": 3, "url": "https://www.technologyreview.com/topic/artificial-intelligence/feed"},
        {"fuente": "Wired", "peso": 2, "url": "https://www.wired.com/feed/tag/ai/latest/rss"},
        {"fuente": "VentureBeat", "peso": 2, "url": "https://venturebeat.com/category/ai/feed/"},
        {"fuente": "Hacker News", "peso": 2, "url": "https://hnrss.org/frontpage?q=AI+OR+LLM+OR+OpenAI+OR+Anthropic"},
        {"fuente": "Google News (EN)", "peso": 1, "url": url_google_news("artificial intelligence", "en-US", "US", desde, hasta)},
        {"fuente": "Google News (ES)", "peso": 1, "url": url_google_news("inteligencia artificial", "es", "ES", desde, hasta)},
    ]


# --------------------------------------------------------------------------
# 2. Lectura de feeds
# --------------------------------------------------------------------------

def leer_feeds(fuentes: list[dict]) -> list[dict]:
    """Lee los feeds uno a uno. Un feed caído no puede tumbar el digest entero."""
    entradas: list[dict] = []
    fallos: list[str] = []

    for fuente in fuentes:
        try:
            feed = feedparser.parse(fuente["url"], agent=AGENTE)
        except Exception as exc:  # noqa: BLE001 - queremos seguir con el resto
            fallos.append(f"{fuente['fuente']}: {exc}")
            continue

        if getattr(feed, "bozo", 0) and not feed.entries:
            fallos.append(f"{fuente['fuente']}: {getattr(feed, 'bozo_exception', 'feed ilegible')}")
            continue

        log(f"  {fuente['fuente']}: {len(feed.entries)} entradas")
        entradas.extend(feed.entries)

    if fallos:
        for fallo in fallos:
            log(f"  AVISO fuente no leída -> {fallo}")
    if not entradas:
        raise SystemExit("Ninguna fuente devolvió entradas: no hay nada que resumir.")

    return entradas


def fecha_de_entrada(entrada) -> datetime | None:
    """feedparser normaliza published/updated a UTC en *_parsed."""
    for campo in ("published_parsed", "updated_parsed"):
        valor = entrada.get(campo)
        if valor:
            try:
                return datetime(*valor[:6], tzinfo=timezone.utc).astimezone(ZONA)
            except (TypeError, ValueError):
                continue
    return None


# --------------------------------------------------------------------------
# 3. Filtrado, deduplicación y ranking
# --------------------------------------------------------------------------

def seleccionar(entradas: list[dict], desde: datetime, hasta: datetime) -> tuple[list[dict], dict]:
    """Filtra por ventana temporal, deduplica entre medios y rankea."""
    por_clave: dict[str, dict] = {}
    diag = {
        "total": 0, "sinLink": 0, "sinFecha": 0,
        "fueraDeRango": 0, "sinHost": 0, "ok": 0,
    }

    for entrada in entradas:
        diag["total"] += 1

        link = entrada.get("link") or entrada.get("id") or ""
        if not link:
            diag["sinLink"] += 1
            continue

        fecha = fecha_de_entrada(entrada)
        if fecha is None:
            diag["sinFecha"] += 1
            continue
        if fecha < desde or fecha > hasta:
            diag["fueraDeRango"] += 1
            continue

        match = re.match(r"^https?://([^/?#]+)", str(link), re.I)
        if not match:
            diag["sinHost"] += 1
            continue
        host = re.sub(r":\d+$", "", match.group(1).lower().removeprefix("www."))
        diag["ok"] += 1

        titulo = limpiar_html(entrada.get("title"))
        fuente = NOMBRES.get(host, host)

        # Google News agrega " - Medio" al final del título; lo separamos.
        if host == "news.google.com":
            partes = titulo.split(" - ")
            if len(partes) > 1:
                fuente = partes.pop().strip()
                titulo = " - ".join(partes).strip()
            else:
                fuente = "Google News"

        clave = " ".join(normalizar(titulo).split(" ")[:9])
        if not clave:
            continue

        extracto = limpiar_html(entrada.get("summary") or entrada.get("description"))
        texto = normalizar(f"{titulo} {extracto}")
        aciertos = sum(1 for k in CLAVES if normalizar(k) in texto)
        peso = PESOS.get(host, 1)

        existente = por_clave.get(clave)
        if existente:
            # La misma noticia en varias fuentes es señal fuerte de importancia.
            existente["apariciones"] += 1
            if peso > existente["peso"]:
                existente.update(peso=peso, titulo=titulo, url=link, fuente=fuente)
            continue

        por_clave[clave] = {
            "titulo": titulo,
            "url": link,
            "fuente": fuente,
            "hora": fecha.strftime("%H:%M"),
            "extracto": extracto[:300],
            "peso": peso,
            "aciertos": aciertos,
            "apariciones": 1,
        }

    candidatos = []
    for c in por_clave.values():
        es_ruido = any(r in c["fuente"].lower() for r in RUIDO)
        c["score"] = c["peso"] * 2 + c["aciertos"] + c["apariciones"] * 2 - (12 if es_ruido else 0)
        if c["score"] > 0:
            candidatos.append(c)

    candidatos.sort(key=lambda c: c["score"], reverse=True)
    return candidatos[:MAX_CANDIDATOS], diag


# --------------------------------------------------------------------------
# 4. Resumen con Claude
# --------------------------------------------------------------------------

SYSTEM = "\n".join([
    "Eres un analista que prepara un briefing diario de noticias de inteligencia artificial para un desarrollador de software.",
    "Recibes una lista de titulares candidatos del día anterior, ya deduplicados.",
    "",
    "Tu tarea: elegir las 10 noticias MÁS IMPORTANTES y resumir cada una.",
    "",
    "Criterios de importancia, en este orden:",
    "1. Impacto real: lanzamientos de modelos, cambios de capacidad, precios, disponibilidad de APIs.",
    "2. Movimientos estructurales del sector: financiación relevante, adquisiciones, regulación, litigios.",
    "3. Avances técnicos o de investigación con consecuencias prácticas.",
    "Penaliza: opinión, listicles, contenido promocional, refritos y notas sin hecho nuevo.",
    "",
    "Reglas de salida:",
    "- Escribe SIEMPRE en español de España.",
    '- "titulo": máximo 80 caracteres, sin comillas ni emojis.',
    '- "resumen": 1 o 2 frases con el hecho concreto y por qué importa. Nada de relleno.',
    '- "url" y "fuente": cópialos EXACTAMENTE del candidato elegido. No inventes URLs.',
    '- "importancia": 1 (menor) a 10 (mayor).',
    "- Ordena el array de mayor a menor importancia.",
    "- Si hay menos de 10 candidatos con valor real, devuelve menos. No rellenes.",
    '- "resumen_global": una frase sobre el tema dominante del día.',
])

ESQUEMA = {
    "type": "object",
    "properties": {
        "titulares": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "titulo": {"type": "string"},
                    "resumen": {"type": "string"},
                    "fuente": {"type": "string"},
                    "url": {"type": "string"},
                    "importancia": {"type": "integer", "enum": list(range(1, 11))},
                },
                "required": ["titulo", "resumen", "fuente", "url", "importancia"],
                "additionalProperties": False,
            },
        },
        "resumen_global": {"type": "string"},
    },
    "required": ["titulares", "resumen_global"],
    "additionalProperties": False,
}


def construir_body(candidatos: list[dict], fecha_texto: str) -> dict:
    listado = "\n\n".join(
        "\n".join(filter(None, [
            f"[{i + 1}] {c['titulo']}",
            f"    fuente: {c['fuente']} | hora: {c['hora']} | apariciones: {c['apariciones']}",
            f"    url: {c['url']}",
            f"    extracto: {c['extracto']}" if c["extracto"] else None,
        ]))
        for i, c in enumerate(candidatos)
    )

    return {
        # Sonnet 5 en vez de Opus: resumir titulares no necesita el modelo más caro.
        "model": MODELO,
        "max_tokens": 16000,
        "system": SYSTEM,
        "messages": [
            {"role": "user", "content": f"Titulares candidatos del {fecha_texto}:\n\n{listado}"}
        ],
        "output_config": {"format": {"type": "json_schema", "schema": ESQUEMA}},
    }


def llamar_claude(body: dict, api_key: str, reintentos: int = 3) -> dict:
    datos = json.dumps(body).encode("utf-8")
    ultimo_error = ""

    for intento in range(1, reintentos + 1):
        peticion = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=datos,
            method="POST",
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
                # La credencial sólo inyecta x-api-key: la versión va aparte.
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(peticion, timeout=300) as respuesta:
                return json.loads(respuesta.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            cuerpo = exc.read().decode("utf-8", "replace")[:500]
            ultimo_error = f"HTTP {exc.code}: {cuerpo}"
            # 4xx que no sea rate limit es un fallo nuestro: no insistimos.
            if exc.code != 429 and exc.code < 500:
                break
        except (urllib.error.URLError, TimeoutError) as exc:
            ultimo_error = str(exc)

        if intento < reintentos:
            espera = 2 ** intento
            log(f"  reintento {intento}/{reintentos - 1} en {espera}s ({ultimo_error})")
            time.sleep(espera)

    raise RuntimeError(f"La API de Anthropic falló: {ultimo_error}")


def extraer_datos(respuesta: dict) -> dict:
    if respuesta.get("stop_reason") == "refusal":
        detalle = (respuesta.get("stop_details") or {}).get("explanation", "sin detalle")
        raise RuntimeError(f"Anthropic rechazó la petición: {detalle}")

    bloque = next((b for b in respuesta.get("content", []) if b.get("type") == "text"), None)
    if not bloque:
        raise RuntimeError(
            "La respuesta no contiene ningún bloque de texto: "
            + json.dumps(respuesta)[:500]
        )

    texto = bloque["text"].strip()
    if texto.startswith("```"):  # por si el modelo envuelve el JSON en un fence
        texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto)

    try:
        return json.loads(texto)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"La respuesta no es JSON válido: {texto[:500]}") from exc


# --------------------------------------------------------------------------
# 5. Mensajes de Telegram
# --------------------------------------------------------------------------

def formatear_mensajes(datos: dict, fecha_texto: str) -> list[str]:
    bloques = [
        f"<b>🤖 Noticias IA — {escapar(fecha_texto)}</b>\n\n"
        f"<i>{escapar(datos.get('resumen_global'))}</i>"
    ]

    for i, n in enumerate(datos.get("titulares") or [], start=1):
        bloques.append("\n".join([
            f"<b>{i}. {escapar(n.get('titulo'))}</b>",
            escapar(n.get("resumen")),
            f"<a href=\"{escapar(n.get('url'))}\">{escapar(n.get('fuente'))}</a>"
            f" · importancia {escapar(n.get('importancia'))}/10",
        ]))

    # Agrupamos bloques enteros sin partir una noticia por la mitad.
    mensajes: list[str] = []
    actual = ""
    for bloque in bloques:
        if actual and len(actual) + 2 + len(bloque) > LIMITE_TELEGRAM:
            mensajes.append(actual)
            actual = bloque
        else:
            actual = f"{actual}\n\n{bloque}" if actual else bloque
    if actual:
        mensajes.append(actual)

    return mensajes


def mensaje_sin_noticias(fecha_texto: str) -> list[str]:
    return [
        f"<b>🤖 Noticias IA — {escapar(fecha_texto)}</b>\n\n"
        "No se encontraron noticias de IA publicadas ese día en las fuentes configuradas."
    ]


def enviar_telegram(mensajes: list[str], token: str, chat_id: str) -> None:
    for i, mensaje in enumerate(mensajes, start=1):
        datos = json.dumps({
            "chat_id": chat_id,
            "text": mensaje,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode("utf-8")

        peticion = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=datos,
            method="POST",
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(peticion, timeout=60) as respuesta:
                respuesta.read()
        except urllib.error.HTTPError as exc:
            cuerpo = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"Telegram devolvió HTTP {exc.code}: {cuerpo}") from exc

        log(f"  mensaje {i}/{len(mensajes)} enviado ({len(mensaje)} caracteres)")
        if i < len(mensajes):
            time.sleep(1)  # el límite de Telegram es ~30 mensajes/segundo


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------

def ventana(modo: str, ahora: datetime) -> tuple[datetime, datetime, datetime, str]:
    """Devuelve (desde, hasta, referencia, texto) según el modo de ventana."""
    if modo == "ultimas24h":
        desde = ahora - timedelta(hours=24)
        return desde, ahora, ahora, f"últimas 24 h hasta {ahora.strftime('%d/%m %H:%M')}"

    ayer = ahora - timedelta(days=1)
    desde = ayer.replace(hour=0, minute=0, second=0, microsecond=0)
    hasta = ayer.replace(hour=23, minute=59, second=59, microsecond=999999)
    return desde, hasta, ayer, fecha_en_espanol(ayer)


def main() -> int:
    parser = argparse.ArgumentParser(description="Digest diario de noticias de IA por Telegram.")
    parser.add_argument(
        "--ventana", choices=["ayer", "ultimas24h"], default="ayer",
        help="'ayer' (día natural anterior, el digest de las 9) o 'ultimas24h' (para probar a cualquier hora).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Salta la comprobación de que sean las %d:00 en la zona configurada." % HORA_DIGEST,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="No llama a Anthropic ni a Telegram: imprime los candidatos seleccionados.",
    )
    parser.add_argument(
        "--marca", metavar="ARCHIVO",
        help="Crea este archivo sólo si el digest llegó a enviarse. Lo usa el "
             "workflow para no repetir el envío si el cron se retrasa.",
    )
    args = parser.parse_args()

    def marcar_enviado() -> None:
        if not args.marca:
            return
        ruta = pathlib.Path(args.marca)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(f"{datetime.now(ZONA).isoformat()}\n", encoding="utf-8")

    ahora = datetime.now(ZONA)

    # El cron de GitHub Actions sólo entiende UTC, así que el workflow dispara a
    # dos horas UTC distintas y es aquí donde se decide cuál corresponde a la
    # hora local pedida. Así el digest sale a la misma hora todo el año, con y
    # sin horario de verano.
    if not args.force and ahora.hour != HORA_DIGEST:
        log(
            f"Son las {ahora:%H:%M} en {ZONA.key} y el digest es a las {HORA_DIGEST}:00. "
            "No toca: salgo sin hacer nada."
        )
        return 0

    desde, hasta, referencia, fecha_texto = ventana(args.ventana, ahora)
    log(f"Digest de: {fecha_texto} ({ZONA.key})")

    log("Leyendo fuentes RSS...")
    entradas = leer_feeds(construir_fuentes(referencia))

    candidatos, diag = seleccionar(entradas, desde, hasta)
    # diagnóstico explícito: un filtrado que colapsa a 0 parece "un día tranquilo".
    log(f"Filtrado: {json.dumps(diag)}")
    log(f"Candidatos seleccionados: {len(candidatos)}")

    if args.dry_run:
        for i, c in enumerate(candidatos[:15], start=1):
            log(f"  [{i}] ({c['score']}) {c['fuente']} — {c['titulo']}")
        log("--dry-run: no se llama a Anthropic ni a Telegram.")
        return 0

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    faltan = [
        nombre for nombre, valor in [
            ("ANTHROPIC_API_KEY", api_key),
            ("TELEGRAM_BOT_TOKEN", token),
            ("TELEGRAM_CHAT_ID", chat_id),
        ] if not valor
    ]
    if faltan:
        log(f"ERROR: faltan variables de entorno: {', '.join(faltan)}")
        return 1

    if not candidatos:
        log("Sin candidatos: aviso por Telegram y termino.")
        enviar_telegram(mensaje_sin_noticias(fecha_texto), token, chat_id)
        marcar_enviado()
        return 0

    log(f"Resumiendo con {MODELO}...")
    respuesta = llamar_claude(construir_body(candidatos, fecha_texto), api_key)
    uso = respuesta.get("usage", {})
    log(
        f"Tokens: entrada {uso.get('input_tokens', '?')} / "
        f"salida {uso.get('output_tokens', '?')} | modelo {respuesta.get('model')}"
    )

    datos = extraer_datos(respuesta)
    mensajes = formatear_mensajes(datos, fecha_texto)
    log(f"Enviando {len(mensajes)} mensaje(s) a Telegram...")
    enviar_telegram(mensajes, token, chat_id)
    marcar_enviado()
    log("Listo.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, SystemExit) as exc:
        if isinstance(exc, SystemExit):
            raise
        log(f"ERROR: {exc}")
        sys.exit(1)
