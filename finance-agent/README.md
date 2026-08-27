# Finance Agent API

Agente conversacional con memoria por hilo y acceso a datos de mercado de Yahoo
Finance. FastAPI + LangChain, con el bucle del agente escrito a mano.

```
POST /chat            → manda un mensaje a un hilo (lo crea si no existe)
GET  /chat/{id}       → devuelve el historial de ese hilo
GET  /health          → liveness
GET  /docs            → OpenAPI interactivo
```

## Levantarlo

```bash
cp .env.example .env       # y completar ANTHROPIC_API_KEY
docker compose up --build
```

```bash
# Turno 1: se crea el hilo y su id viene en la respuesta
curl -s -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"message":"¿a cuánto cotiza Apple?"}'

# Turno 2: mismo id, el agente recuerda el contexto
curl -s -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"conversation_id":"<id>","message":"¿y comparado con Microsoft?"}'

# Historial
curl -s localhost:8000/chat/<id>
curl -s 'localhost:8000/chat/<id>?include_tools=true'   # con los pasos internos
```

Sin Docker:

```bash
pip install -r requirements-dev.txt
export ANTHROPIC_API_KEY=...
uvicorn app.main:app --reload
pytest
```

## Cómo está armado

```
app/
├── main.py           arranque, middleware de correlación, manejo de errores
├── config.py         settings desde el entorno (12-factor)
├── schemas.py        contratos HTTP, separados de los modelos internos
├── api/routes.py     los dos endpoints
├── agent/
│   ├── loop.py       ← el bucle del agente, explícito
│   ├── service.py    coordina memoria + bucle
│   ├── history.py    recorte del historial por tokens
│   ├── locks.py      serialización por conversación
│   └── prompts.py    prompt del sistema
├── memory/
│   ├── base.py       protocolo ConversationStore
│   ├── sqlite_store.py   implementación persistente
│   └── memory_store.py   implementación para tests
└── tools/
    └── yahoo_finance.py  las 4 tools sobre yfinance
```

El flujo de un `POST /chat`:

```
POST /chat {conversation_id, message}
  └─ lock del hilo (dos pedidos al mismo id no se intercalan)
      ├─ leer historial de SQLite
      ├─ recortar por tokens + anteponer el system prompt
      ├─ agent_loop:
      │     ┌──────────────────────────────────────────┐
      │     │ modelo.invoke(conversación)              │
      │     │   ¿pidió tools? ──no──→ ésa es la respuesta
      │     │        │ sí                              │
      │     │        ▼                                 │
      │     │ ejecutarlas en paralelo → ToolMessage    │
      │     └──────────┬───────────────────────────────┘
      │                └─ hasta max_iterations, y ahí cierre forzado
      └─ persistir el turno entero (pregunta + todo lo que produjo)
```

## Decisiones de diseño

### El bucle es propio, no `AgentExecutor`

Se pidió explícitamente. `app/agent/loop.py` implementa el ciclo completo y
deja a la vista lo que un framework esconde:

- **Tope de iteraciones.** Un modelo que insiste en llamar a una herramienta
  que siempre falla, sin tope, gasta plata hasta que alguien lo mata a mano.
- **Cierre forzado.** Al agotarse las vueltas se hace una última llamada *sin
  herramientas*. Sirve para dos cosas: que el usuario reciba una respuesta en
  prosa, y que el historial no quede terminado en un `ToolMessage` — un estado
  que la API del proveedor rechaza en el turno siguiente.
- **Los errores de herramienta vuelven al modelo, no al usuario.** Un ticker
  mal escrito llega como `{"error": ..., "code": "symbol_not_found"}` y el
  modelo puede buscar el símbolo correcto y reintentar. Sólo escala como error
  HTTP lo que impide seguir.
- **Llamadas de una misma tanda en paralelo.** Cuando el modelo pide dos
  cotizaciones para compararlas, el turno tarda lo que la más lenta y no la
  suma.
- **Traza de cada paso**, devuelta en la respuesta: qué herramienta, con qué
  argumentos, cuánto tardó, si salió bien. Es lo que permite auditar de dónde
  salió cada número que el agente afirma.

El bucle se tipa contra un `Protocol` (`SupportsToolCalling`) y no contra
`ChatAnthropic`: por eso los tests corren con un modelo guionado, sin red y sin
gastar tokens.

### La memoria es una interfaz, la implementación es SQLite

`ConversationStore` es un `Protocol`; ni la API ni el agente saben que abajo
hay SQLite. Se eligió SQLite con volumen montado porque da durabilidad real
—las conversaciones sobreviven a un reinicio del contenedor— sin sumar otro
servicio al compose.

Se persisten los mensajes con la representación canónica de LangChain
(`messages_to_dict`). Guardar sólo `{role, content}` parece suficiente hasta que
el hilo tiene una llamada a herramienta: si se pierden los `tool_calls` de un
`AIMessage` o el `tool_call_id` de su `ToolMessage`, la conversación deja de
ser válida y el turno siguiente falla con un 400.

El turno se guarda **en una sola operación atómica**: la pregunta y todo lo que
produjo entran juntas o no entra nada. Un turno a medias corrompe el hilo para
siempre.

### Concurrencia

Un lock por `conversation_id` cubre el ciclo entero —leer historial, razonar,
escribir—, no sólo la escritura: si cubriera nada más el INSERT, dos pedidos
simultáneos igualmente habrían razonado sobre el mismo historial viejo y uno de
los dos turnos se perdería. Conversaciones distintas siguen corriendo en
paralelo. El registro de locks se limpia solo (`app/agent/locks.py`).

### El historial se recorta por tokens, no por cantidad de mensajes

Un resultado de `get_price_history` pesa lo que veinte turnos de charla. El
recorte usa `trim_messages` con `start_on="human"`, que garantiza que el corte
cae en un límite de turno y nunca deja un `ToolMessage` huérfano.

### Las herramientas

Cuatro, sobre `yfinance`: `search_symbol`, `get_quote`, `get_price_history`,
`get_fundamentals`. `search_symbol` existe porque la gente dice "Apple", no
"AAPL".

- yfinance es síncrono y hace red: cada llamada va a un thread con timeout, para
  no congelar el event loop y con él a todos los demás pedidos del proceso.
- Las respuestas vienen **recortadas**. Un `history` de un año son 250 filas que
  el modelo no necesita y que se pagan por token en *cada turno siguiente*,
  porque quedan en el historial. Se manda un resumen estadístico y una serie
  submuestreada que siempre incluye la última observación.
- `NaN` e infinito, que son valores normales en pandas, se convierten a `null`:
  `json.dumps` los escribe como `NaN`, que es JSON inválido y rompe al cliente.

### Operación

- **Logs JSON de una línea por evento**, con `request_id` y `conversation_id` en
  cada uno. El `request_id` se toma del header `X-Request-ID` si el cliente lo
  manda y si no se genera; vuelve siempre en la respuesta, para que un usuario
  que reporta un error tenga algo que darnos para encontrarlo.
- **Nunca se filtra un traceback** al cliente: recibe un `request_id` y un
  código, y el detalle queda en los logs contra ese mismo id.
- **Falla al arrancar** si falta `ANTHROPIC_API_KEY`, en vez de arrancar bien y
  romper con el primer usuario.
- **Ids validados** contra `^[A-Za-z0-9_-]{1,64}$`: entran en claves de caché y
  en nombres de lock.
- Imagen multi-stage, usuario sin privilegios, healthcheck sin instalar `curl`.

### Sobre el alcance del asistente

El prompt del sistema le prohíbe recomendar comprar o vender y proyectar
precios. Un asistente que consulta cotizaciones y encima aconseja es un
problema regulatorio, no una feature.

## Tests

```bash
pytest              # 51 tests, sin red y sin gastar tokens
```

El modelo es un doble que devuelve un guion prefijado, y `yfinance` está
mockeado. Un test que dependa del mercado real falla los sábados y cuando Yahoo
rate-limitea; lo que se verifica acá es nuestro código, no que Yahoo funcione.

Los casos que más importan son los de error: herramienta que revienta,
herramienta que no existe, timeout, tope de iteraciones, y el cierre forzado
fallando también.

Los tests de memoria corren **contra las dos implementaciones** del store: si
la de SQLite y la de memoria se comportaran distinto, los tests rápidos
dejarían de decir algo sobre lo que corre en producción.

## Qué haría distinto en producción

Lo que este MVP deliberadamente no resuelve:

| Límite | Qué haría |
|---|---|
| SQLite es de un solo proceso escritor; el Dockerfile fija `--workers 1` | Postgres detrás del mismo `ConversationStore`, y ahí sí varias réplicas |
| Los locks son de proceso: con dos réplicas dejan de garantizar nada | Lock distribuido (Redis) o particionar los hilos por réplica |
| La caché de cotizaciones es un dict por proceso | Redis, si molesta la duplicación entre réplicas |
| No hay autenticación: cualquiera puede leer cualquier hilo con su id | API key por cliente y hilos con dueño |
| No hay rate limiting; cada `POST /chat` cuesta tokens reales | Cuota por cliente antes de llegar al modelo |
| Sin streaming: la respuesta llega entera al final | SSE, que es donde más se nota la latencia de un agente |
| Sin idempotencia: un reintento del cliente duplica el turno | `Idempotency-Key` sobre el `POST` |
| Los hilos crecen para siempre | Retención por antigüedad, y resumen de los turnos viejos en vez de recorte |

## Nota sobre redes con proxy

`yfinance` usa `curl_cffi` para imitar el TLS de un navegador, y un proxy
corporativo que reinspecciona TLS lo rechaza (`Recv failure: Connection reset`).
`YF_DISABLE_CURL_CFFI=1` lo hace caer al backend de `requests`, que sí respeta
`HTTPS_PROXY` y `REQUESTS_CA_BUNDLE`.
