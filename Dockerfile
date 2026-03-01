# ─────────────────────────────────────────────────────────────────────────────
#  Dockerfile — Voice AI Agent (webhook + worker service)
#
#  Build:  docker compose build
#  Run:    docker compose up -d
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps needed by sentence-transformers and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (better Docker layer caching)
# If requirements.txt hasn't changed, this layer is reused on rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories inside the container
# These get overridden by the volume mount in docker-compose
RUN mkdir -p /app/data/chromadb /app/logs

# The webhook service listens on this port
EXPOSE 8000

# Default entry point — starts the FastAPI webhook server
CMD ["python", "-m", "uvicorn", "services.webhook.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]