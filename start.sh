#!/usr/bin/env bash
# Start script for Render.com

set -o errexit  # Exit on error

echo "🚀 Starting FastAPI application..."

# Run database migrations if needed
# alembic upgrade head

# Start uvicorn with production settings
uvicorn src.api_v2:app \
  --host 0.0.0.0 \
  --port ${PORT:-10000} \
  --workers 2 \
  --log-level info
