from typing import Optional,cast
from uuid import uuid4
from prefect.client.schemas import FlowRun
from prefect.exceptions import ObjectNotFound
from fastapi import (
    APIRouter,
    status,
    HTTPException
)
from core.lifespan import DEPLOY_IDENTIFIER
from pydantic import BaseModel, Field
from loguru import logger
# from flow.video_processing import video_processing_flow
import os
from prefect.deployments import run_deployment

router = APIRouter(prefix="/uploads", tags=["uploads"])



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



class VideoObject(BaseModel):
    video_id: str
    video_url: str

class UploadRequest(BaseModel):
    videos: list[VideoObject] = Field(..., description="list of uploading videos, in the format of (video_id, video_s3_url)")
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

    video_files = request.videos
    video_files_tuple = []
    for vid_file in video_files:
        video_files_tuple.append(
            (vid_file.video_id, vid_file.video_url)
        )
    try:
        flow_run: FlowRun = await run_deployment( #type:ignore
            name=DEPLOY_IDENTIFIER,
            parameters={
                "video_files": video_files_tuple,
                "user_id": request.user_id,
                "run_id": run_id,
            },
            flow_run_name=run_id,
            idempotency_key=run_id,
            timeout=0
        )
        logger.info(
            f"Triggered Prefect deployment '{DEPLOY_IDENTIFIER}' "
            f"with flow_run_id={flow_run.id}, run_id={run_id}"
        )
    except ObjectNotFound as exc:
        logger.exception(f"Deployment '{DEPLOY_IDENTIFIER}' not found")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prefect deployment '{DEPLOY_IDENTIFIER}' is not registered. "
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
        video_names=[video_info.video_id for  video_info in request.videos],
        status=flow_run.state.type.value if flow_run.state else "SCHEDULED",
        message="Video processing deployment submitted successfully",
        tracking_url=f"/api/management/videos/{run_id}/status",
        deployment_name=DEPLOY_IDENTIFIER,
    )