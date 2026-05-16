"""
Celery application factory.

Celery uses Redis as both the message broker and result backend.
Workers are started separately:
    cd backend && uv run celery -A app.worker.celery_app worker --loglevel=info --concurrency=4
"""

import os
from celery import Celery

# Fall back to localhost if REDIS_URL is not set (dev-friendly default)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "backend",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Reliability
    task_acks_late=True,          # acknowledge *after* task completes (no lost jobs on crash)
    worker_prefetch_multiplier=1, # one task per worker at a time — fair distribution
    # Result TTL
    result_expires=3600,          # keep results 1 hour
    # Retry default
    task_max_retries=3,
)
