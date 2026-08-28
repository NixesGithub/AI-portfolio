# syntax=docker/dockerfile:1

# Dos etapas: las dependencias se instalan en un venv que se copia a una imagen
# final sin toolchain de compilación. La imagen que se despliega no lleva gcc ni
# los headers de Python, que es superficie de ataque que no hace falta en runtime.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Sólo requirements primero: mientras no cambien, esta capa se reusa y las
# builds siguientes no vuelven a bajar pandas.
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Usuario sin privilegios: si alguien escapa del proceso, no es root.
RUN useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv
COPY --chown=appuser:appuser app ./app

# El volumen de datos tiene que existir y pertenecerle al usuario antes de
# bajar de privilegios.
RUN mkdir -p /srv/data && chown appuser:appuser /srv/data
VOLUME ["/srv/data"]

USER appuser

ENV DATABASE_URL="sqlite:///./data/conversations.db"

EXPOSE 8000

# El healthcheck usa la stdlib: no instalamos curl sólo para esto.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# Un solo worker a propósito: el store es SQLite y los locks por conversación
# son de proceso. Escalar a varios workers requiere primero mover el store a
# Postgres (ver README, "Qué haría distinto en producción").
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
