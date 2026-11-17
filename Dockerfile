# ── PipelineDoctor FastAPI Control Plane ────────────────────────────────────
FROM python:3.11-slim

# Install system utilities: patch, git, curl, and docker CLI for container sandbox execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    patch \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install hatchling build backend
RUN pip install --no-cache-dir hatchling

# Copy project definition and README for editable install
COPY pyproject.toml README.md ./

# Install package dependencies
RUN pip install --no-cache-dir .

# Copy application source code
COPY pipelinedoctor/ ./pipelinedoctor/
COPY scripts/ ./scripts/

# Expose FastAPI HTTP control plane
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch uvicorn server
CMD ["uvicorn", "pipelinedoctor.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
