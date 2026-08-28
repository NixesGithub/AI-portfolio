#!/usr/bin/env bash
# Conversación de ejemplo contra la API (por defecto, http://localhost:8000).
# Demuestra lo que pide el enunciado: memoria por id y consulta a Yahoo Finance.
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
CONV="demo-$(date +%s)"
AUTH=()
[[ -n "${API_KEY:-}" ]] && AUTH=(-H "X-API-Key: ${API_KEY}")

say() { printf '\n\033[1m%s\033[0m\n' "$1"; }

chat() {
  curl -sS -X POST "${BASE_URL}/chat" \
    -H 'Content-Type: application/json' "${AUTH[@]}" \
    -d "{\"conversation_id\":\"${CONV}\",\"message\":$(printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}" \
    | python3 -m json.tool
}

say "Salud del servicio"
curl -sS "${BASE_URL}/health/ready" | python3 -m json.tool

say "1) Primer mensaje (crea el hilo ${CONV})"
chat "¿A cuánto cotiza Apple ahora mismo?"

say "2) Segundo mensaje: referencia a lo anterior, sin repetir el ticker"
chat "¿Y cómo le ha ido en el último mes?"

say "3) Historial del hilo"
curl -sS "${AUTH[@]}" "${BASE_URL}/chat/${CONV}" | python3 -m json.tool

say "4) Historial con la traza de herramientas"
curl -sS "${AUTH[@]}" "${BASE_URL}/chat/${CONV}?include_tools=true" | python3 -m json.tool

say "5) Otro hilo no ve nada de lo anterior"
curl -sS -X POST "${BASE_URL}/chat" -H 'Content-Type: application/json' "${AUTH[@]}" \
  -d '{"conversation_id":"demo-otro-hilo","message":"¿De qué estábamos hablando?"}' \
  | python3 -m json.tool
