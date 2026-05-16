"""
Celery tasks for video ingestion and deletion.

Each video upload dispatches one `ingest_video_task` per video so that:
  - Videos are processed concurrently (controlled by --concurrency flag on worker).
  - Each video has its own independent retry budget (max 3 retries, exponential back-off).
  - The FastAPI upload endpoint returns immediately; heavy I/O happens off the event loop.

Task flow:
  FastAPI upload endpoint
      └─► ingest_video_task.delay(video_id, video_url, user_id)
              └─► HTTP POST → video_pipeline ingestion service
                      └─► pipeline calls back /api/ingestion/service/status/{video_id}
                              └─► Socket.IO → frontend (existing flow, unchanged)
"""

import logging
import time

import requests as _requests  # sync requests is fine inside Celery (no async event loop)

from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Configuration  (read from env via os.getenv so
# the worker process also respects .env values)
# ──────────────────────────────────────────────
import os

INGESTION_SERVICE_URL = os.getenv(
    "INGESTION_SERVICE_URL", "http://100.113.186.28:8050"
)
INGESTION_CANCEL_URL = os.getenv(
    "INGESTION_CANCEL_URL", "http://100.113.186.28:8000"
)

# ──────────────────────────────────────────────
# Tasks
# ──────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="app.worker.tasks.ingest_video",
    max_retries=3,
    default_retry_delay=10,  # first retry after 10 s
)
def ingest_video_task(self, video_id: str, video_url: str, user_id: str) -> dict:
    """
    Dispatch a single video to the ingestion pipeline service.

    This task is intentionally small: it just makes the HTTP call.
    The ingestion service handles the heavy ML work and calls back to the
    backend when done (`/api/ingestion/service/status/{video_id}`).

    Retry strategy: 3 retries with exponential back-off (10s, 20s, 40s).
    """
    logger.info(
        "📤 [ingest_video] video_id=%s user_id=%s attempt=%s",
        video_id,
        user_id,
        self.request.retries + 1,
    )

    try:
        response = _requests.post(
            f"{INGESTION_SERVICE_URL}/uploads/",
            json={"videos": [{"video_id": video_id, "video_url": video_url}], "user_id": user_id},
            timeout=30,
        )
        response.raise_for_status()
        logger.info("✅ [ingest_video] Accepted by pipeline: video_id=%s", video_id)
        return {"video_id": video_id, "status": "accepted", "code": response.status_code}

    except _requests.Timeout as exc:
        logger.warning(
            "⚠️  [ingest_video] Timeout for video_id=%s (attempt %s), retrying...",
            video_id,
            self.request.retries + 1,
        )
        # Exponential back-off: 10s, 20s, 40s
        raise self.retry(exc=exc, countdown=10 * (2 ** self.request.retries))

    except _requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "N/A"
        if status_code and 400 <= status_code < 500:
            # 4xx: client errors — don't retry (video_id wrong, auth, etc.)
            logger.error(
                "❌ [ingest_video] 4xx error for video_id=%s (status %s), not retrying.",
                video_id,
                status_code,
            )
            return {"video_id": video_id, "status": "failed", "code": status_code}
        logger.warning(
            "⚠️  [ingest_video] HTTP %s for video_id=%s, retrying...",
            status_code,
            video_id,
        )
        raise self.retry(exc=exc, countdown=10 * (2 ** self.request.retries))

    except Exception as exc:
        logger.error(
            "❌ [ingest_video] Unexpected error for video_id=%s: %s",
            video_id,
            exc,
        )
        raise self.retry(exc=exc, countdown=10 * (2 ** self.request.retries))


@celery_app.task(
    bind=True,
    name="app.worker.tasks.cancel_ingestion",
    max_retries=2,
    default_retry_delay=5,
)
def cancel_ingestion_task(self, video_id: str) -> dict:
    """
    Tell the ingestion pipeline to cancel/clean up a video run.
    Called when the user deletes a video mid-ingestion.
    """
    logger.info("🛑 [cancel_ingestion] video_id=%s", video_id)

    try:
        response = _requests.post(
            f"{INGESTION_CANCEL_URL}/management/runs/{video_id}/cancel",
            json={"video_id": video_id},
            timeout=15,
        )
        response.raise_for_status()
        logger.info("✅ [cancel_ingestion] Cancelled: video_id=%s", video_id)
        return {"video_id": video_id, "status": "cancelled"}

    except Exception as exc:
        logger.warning(
            "⚠️  [cancel_ingestion] Failed for video_id=%s: %s — retrying",
            video_id,
            exc,
        )
        raise self.retry(exc=exc, countdown=5 * (2 ** self.request.retries))
