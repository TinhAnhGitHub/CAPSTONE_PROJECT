from typing import Optional,cast
from uuid import uuid4
from prefect.client.schemas import FlowRun
from prefect.exceptions import ObjectNotFound
from fastapi import (
    APIRouter,
    status,
    HTTPException
)
from pydantic import BaseModel, Field
from loguru import logger
# from flow.video_processing import video_processing_flow
import os
from prefect.deployments import run_deployment

router = APIRouter(prefix="/uploads", tags=["uploads"])


VIDEO_PROCESSING_DEPLOYMENT = cast(str, os.getenv('VIDEO_PROCESSING_DEPLOYMENT'))

class UploadResponse(BaseModel):
    """Response after successful upload."""
    run_id: str
    flow_run_id: str
    video_count: int
    video_names: list[str]
    status: str
    message: str
    deployment_name: str
    tracking_url: Optional[str] = None



class UploadRequest(BaseModel):
    videos: list[tuple[str,str]] = Field(..., description="list of uploading videos, in the format of (video_id, video_s3_url)")
    user_id: str

@router.post(
    "/",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload videos and start processing",
    description="Upload one or more videos and trigger the Prefect deployment for processing",
)
async def upload_videos(
    request: UploadRequest,
) -> UploadResponse:
    
    logger.info(f"Calling video_processing_flow directly")
    run_id = str(uuid4())
    try:
        flow_run: FlowRun = await run_deployment( #type:ignore
            name=VIDEO_PROCESSING_DEPLOYMENT,
            parameters={
                "video_files": request.videos,
                "user_id": request.user_id,
                "run_id": run_id,
            },
            timeout=0
        )
        logger.info(
            f"Triggered Prefect deployment '{VIDEO_PROCESSING_DEPLOYMENT}' "
            f"with flow_run_id={flow_run.id}, run_id={run_id}"
        )
    except ObjectNotFound as exc:
        logger.exception(f"Deployment '{VIDEO_PROCESSING_DEPLOYMENT}' not found")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prefect deployment '{VIDEO_PROCESSING_DEPLOYMENT}' is not registered. "
                   f"Please run 'prefect deploy --name primary-gpu' first.",
        ) from exc
    except Exception as exc:
        logger.exception("Unable to trigger Prefect deployment")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to trigger Prefect deployment",
        ) from exc
        
    return UploadResponse(
        run_id=run_id,
        flow_run_id=str(flow_run.id),
        video_count=len(request.videos),
        video_names=[file_name or "unknown" for _, file_name in request.videos],
        status=flow_run.state.type.value if flow_run.state else "SCHEDULED",
        message="Video processing deployment submitted successfully",
        tracking_url=f"/api/management/videos/{run_id}/status",
        deployment_name=VIDEO_PROCESSING_DEPLOYMENT,
    )