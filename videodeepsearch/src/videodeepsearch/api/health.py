from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "/status",
    summary="Check readiness of all clients and models."
)
async def status(request: Request):
    """Check all initialized clients in app.state."""
    state = request.app.state

    return {
        "postgres": bool(getattr(state, "postgres_client", None)),
        "minio": bool(getattr(state, "minio_client", None)),
        "qdrant_image": bool(getattr(state, "image_qdrant_client", None)),
        "qdrant_segment": bool(getattr(state, "segment_qdrant_client", None)),
        "qdrant_audio": bool(getattr(state, "audio_qdrant_client", None)),
        "elasticsearch": bool(getattr(state, "es_ocr_client", None)),
        "arangodb": bool(getattr(state, "arango_db", None)),
        "qwenvl": bool(getattr(state, "qwenvl_client", None)),
        "mmbert": bool(getattr(state, "mmbert_client", None)),
        "splade": bool(getattr(state, "splade_client", None)),
        "models": list(getattr(state, "models", {}).keys()),
        "worker_models": list(getattr(state, "worker_models", {}).keys()),
    }


@router.get(
    "/ready",
    summary="Overall readiness signal."
)
async def readiness(request: Request):
    """Check if all required services are ready."""
    state = request.app.state

    checks = {
        "postgres": bool(getattr(state, "postgres_client", None)),
        "minio": bool(getattr(state, "minio_client", None)),
        "qdrant": all([
            getattr(state, "image_qdrant_client", None),
            getattr(state, "segment_qdrant_client", None),
            getattr(state, "audio_qdrant_client", None),
        ]),
        "elasticsearch": bool(getattr(state, "es_ocr_client", None)),
        "arangodb": bool(getattr(state, "arango_db", None)),
        "inference": all([
            getattr(state, "qwenvl_client", None),
            getattr(state, "mmbert_client", None),
            getattr(state, "splade_client", None),
        ]),
        "models": bool(getattr(state, "models", None)),
        "worker_models": bool(getattr(state, "worker_models", None)),
    }

    ready = all(checks.values())
    return {"ready": ready, "checks": checks}