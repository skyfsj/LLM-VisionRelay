FROM python:3.12-slim

# Build-time proxy args (used only during build; not persisted at runtime).
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime deps first for better layer caching.
COPY pyproject.toml README.md ./
COPY llm_visionrelay ./llm_visionrelay
RUN pip install --no-cache-dir .

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8080

CMD ["python", "-m", "llm_visionrelay", "--host", "0.0.0.0", "--port", "8080", "--cache-dir", "/data"]
