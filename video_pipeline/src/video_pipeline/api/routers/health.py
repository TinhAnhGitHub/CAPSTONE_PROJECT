from __future__ import annotations

from fastapi import APIRouter
from prefect.client.orchestration import get_client

from video_pipeline.api.lifespan import DEPLOY_IDENTIFIER

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health() -> dict:
    return {"status": "ok", "service": "video-pipeline-api"}


@router.get("/prefect")
async def prefect_health() -> dict:
    """Check connectivity to the Prefect server and verify the deployment exists."""
    try:
        async with get_client() as client:
            await client.api_healthcheck()
            deployment = await client.read_deployment_by_name(DEPLOY_IDENTIFIER)
        return {
            "status": "ok",
            "prefect": "connected",
            "deployment": DEPLOY_IDENTIFIER,
            "deployment_id": str(deployment.id),
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "prefect": str(exc),
            "deployment": DEPLOY_IDENTIFIER,
        }
