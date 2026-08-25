# AI-portfolio

This is a portfolio project demonstrating my AI skills.

## Noticias IA diarias

Un resumen de las noticias de IA del día anterior, en Telegram, todas las mañanas
a las **9:00 (hora de Madrid)**.

Es el port de un workflow de [n8n](https://github.com/NixesGithub/notic-ia) que
antes corría en local con Docker. La lógica es la misma; lo que cambia es dónde
se ejecuta: ahora corre en los runners de GitHub Actions, así que **da igual si
el portátil está encendido, suspendido o apagado**. La versión de n8n no
recuperaba las ejecuciones perdidas, así que un día con el equipo apagado era un
día sin digest.

```
09:00 Europe/Madrid (GitHub Actions)
   └─ 9 fuentes RSS (~280 noticias)
      └─ Filtrar por el día anterior, deduplicar entre medios y rankear → top 40
         └─ Resumir con Claude (salida JSON validada contra esquema) → top 10
            └─ Enviar a Telegram (HTML, troceado a 3800 caracteres)
```

| Archivo | Qué es |
|---|---|
| `.github/workflows/noticias-ia.yml` | El cron y el job |
| `scripts/noticias_ia.py` | El pipeline completo |
| `requirements.txt` | `feedparser` (lo demás es librería estándar) |

### Puesta en marcha

Hacen falta tres secretos en **Settings → Secrets and variables → Actions**:

| Secreto | De dónde sale |
|---|---|
| `ANTHROPIC_API_KEY` | https://console.anthropic.com → *API Keys* |
| `TELEGRAM_BOT_TOKEN` | El token que da **@BotFather** con `/newbot` |
| `TELEGRAM_CHAT_ID` | Escribile al bot y mirá `result[0].message.chat.id` en `https://api.telegram.org/bot<TOKEN>/getUpdates` |

Ojo: en la versión de n8n el token del bot vivía en una credencial y sólo el
chat id estaba en `.env`. Aquí los tres son secretos del repo.

### Probarlo sin esperar a mañana

Desde **Actions → Noticias IA diarias → Run workflow**. La ejecución manual salta
la comprobación de hora, y tiene dos opciones:

- **ventana**: `ultimas24h` (por defecto en manual) mira las últimas 24 h desde
  este momento; `ayer` reproduce exactamente lo que haría el cron.
- **dry_run**: lista los candidatos y termina, sin gastar API de Anthropic ni
  mandar nada a Telegram.

En local es lo mismo:

```bash
pip install -r requirements.txt
python scripts/noticias_ia.py --dry-run --force              # sólo ver candidatos
python scripts/noticias_ia.py --force --ventana ultimas24h   # envío real
```

### Cómo se decide "lo más importante"

Dos capas. La primera es determinista, dentro del script:

- **Filtro de fecha**: sólo noticias publicadas entre las 00:00 y las 23:59 del
  día anterior, hora de España.
- **Deduplicación**: se normaliza el título (sin acentos ni puntuación) y se
  agrupan las 9 primeras palabras. Si la misma noticia sale en 3 medios cuenta
  como una sola, pero **suma puntos**: aparecer en varios sitios es la mejor
  señal de importancia que hay.
- **Puntuación**: `peso_fuente × 2 + palabras_clave + apariciones × 2`,
  penalizando 12 puntos el contenido bursátil sindicado (Motley Fool, Zacks,
  Benzinga…) que inunda las búsquedas de "IA".
- Se queda con los **40 mejores**.

La segunda capa es **Claude** (`claude-sonnet-5`), que recibe esos 40 titulares,
elige el top 10 real, lo resume en español y le pone una nota de importancia.
La salida va validada contra un esquema JSON, así que siempre es parseable.

Coste: unos 40 titulares de entrada y ~1.500 tokens de salida al día, del orden
de **céntimos**. Los minutos de GitHub Actions son gratis en repos públicos.

### Detalles que no son obvios

**El cron de GitHub Actions sólo entiende UTC, y Madrid cambia con el horario de
verano.** Un cron fijo daría las 9:00 media parte del año y las 8:00 o las 10:00
la otra. Por eso el workflow dispara a **las 07:00 y las 08:00 UTC** y es el
script el que comprueba si son las 9 en `Europe/Madrid`; la ejecución que no
toca sale sin hacer nada. Si cambiás `DIGEST_TZ` o `DIGEST_HOUR` en el workflow,
acordate de mover también esas dos horas UTC del `cron`.

**GitHub no garantiza la puntualidad de los crons.** En horas punta la ejecución
puede retrasarse bastantes minutos. Como la comprobación es sobre la *hora* y no
sobre el minuto exacto, el digest sale igual. Si el retraso fuese tan grande que
las dos ejecuciones del día cayesen dentro de la hora del digest, una marca
guardada en la caché de Actions (`noticias-ia-enviado-<fecha>`) evita que llegue
por duplicado.

**Los feeds sólo guardan entre 10 y 20 noticias.** Ejecutarlo por la tarde da
menos resultados: las de ayer ya se cayeron de la ventana. Por eso las URLs de
Google News se generan con `after:`/`before:` acotando el día anterior — sin eso
Google devuelve casi sólo noticias del día en curso.

**Una fuente caída no tumba el digest.** Cada feed se lee por separado y los
fallos se registran como aviso. El script sí aborta si *ninguna* fuente responde,
que es un fallo real y no un día tranquilo.

**El filtrado emite un `diagnostico`** (`{total, sinLink, sinFecha, fueraDeRango,
sinHost, ok}`) en el log del job, precisamente para que un colapso silencioso del
filtrado se vea en vez de parecer que no hubo noticias.

**GitHub desactiva los workflows programados en repos sin actividad durante 60
días.** Avisa por email; se reactivan desde la pestaña Actions.

### Añadir o quitar fuentes

En `scripts/noticias_ia.py`, la lista `construir_fuentes()` (con el peso de cada
medio) y los diccionarios `PESOS` / `NOMBRES`, que van por host. Si un medio no
tiene RSS, probá con una búsqueda de Google News acotada a ese dominio:

```
https://news.google.com/rss/search?q=site:ejemplo.com+AI&hl=es&gl=ES&ceid=ES:es
```
