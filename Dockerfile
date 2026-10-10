# SwarmSight AI: web service (Hugging Face Spaces "docker" SDK or any container host)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    SWARMSIGHT_RUNTIME=/tmp/swarmsight SWARMSIGHT_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app swarmsight swarmsight
COPY --chown=app web web
COPY --chown=app models models
COPY --chown=app demo demo
COPY --chown=app docs docs
COPY --chown=app datasets/tashkent datasets/tashkent
COPY --chown=app benchmarks/results benchmarks/results

USER app
EXPOSE 7860
CMD ["uvicorn", "swarmsight.server:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
