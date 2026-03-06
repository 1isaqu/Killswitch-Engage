FROM python:3.11-slim

# System dependencies for asyncpg and lightfm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cacheable step)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy application source only (data/, models/, .env excluded via .dockerignore)
COPY src/ ./src/
COPY backend/ ./backend/

# Ensure src is importable as a package
RUN touch src/__init__.py 2>/dev/null || true

# Non-root user for principle of least privilege (SECURITY_CHECKLIST § 7)
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

EXPOSE 8000

# Entry point uses the refactored app factory
CMD ["uvicorn", "src.backend.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
