FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/

RUN python backend/download_model.py
RUN mkdir -p backend/services && \
    cp /app/hybrid_personality_system.pkl /app/backend/services/hybrid_personality_system.pkl

RUN mkdir -p logs models

ENV PYTHONPATH=/app

ENV ENVIRONMENT=production \
    DEBUG=False \
    PYTHONUNBUFFERED=1

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
  CMD curl -f http://localhost:7860/ || exit 1

CMD ["python", "-m", "uvicorn", "backend.main_optimized:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
