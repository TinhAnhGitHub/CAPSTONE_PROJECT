from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from loguru import logger
from prefect.client.schemas import FlowRun
from prefect.deployments import run_deployment
from prefect.exceptions import ObjectNotFound
from pydantic import BaseModel, Field

from video_pipeline.api.lifespan import DEPLOY_IDENTIFIER

router = APIRouter(prefix="/uploads", tags=["uploads"])


class VideoObject(BaseModel):
    video_id: str
    video_url: str = Field(..., description="S3 URL of the video, e.g. s3://bucket/video.mp4")


class UploadRequest(BaseModel):
    videos: list[VideoObject] = Field(..., description="One or more videos to process")
    user_id: str
    # tracker_url: str = Field(
    #     default="http://100.120.22.90:8010",
    #     description="Optional HTTP endpoint for pipeline progress callbacks",
    # )


class VideoFlowResponse(BaseModel):
    video_id: str
    flow_run_id: str
    state: str


class UploadResponse(BaseModel):
    run_id: str
    user_id: str
    video_count: int
    results: list[VideoFlowResponse]
    status: str
    message: str
    deployment_name: str


@router.post(
    "/",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit videos for processing",
    description=(
        "Trigger the Prefect deployment for each video and return immediately. "
        "Processing happens asynchronously in the Prefect worker pool."
    ),
)
async def upload_videos(request: UploadRequest) -> UploadResponse:
    run_id = str(uuid4())
    results: list[VideoFlowResponse] = []

    for video in request.videos:
        idempotency_key = f"{run_id}-{video.video_id}"
        try:
            flow_run: FlowRun = await run_deployment(  # type: ignore
                name=DEPLOY_IDENTIFIER,
                parameters={
                    "video_id": video.video_id,
                    "user_id": request.user_id,
                    "video_file_path": video.video_url,
                    "tracker_url": "http://100.120.22.90:8010",
                },
                flow_run_name=idempotency_key,
                idempotency_key=idempotency_key,
                timeout=0, 
            )
            logger.info(
                f"[upload] Triggered '{DEPLOY_IDENTIFIER}' | "
                f"video_id={video.video_id} flow_run_id={flow_run.id}"
            )
            results.append(
                VideoFlowResponse(
                    video_id=video.video_id,
                    flow_run_id=str(flow_run.id),
                    state=flow_run.state.type.value if flow_run.state else "SCHEDULED",
                )
            )
        except ObjectNotFound as exc:
            logger.exception(f"Deployment '{DEPLOY_IDENTIFIER}' not found")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Prefect deployment '{DEPLOY_IDENTIFIER}' is not registered. "
                    "Run 'prefect deploy --name poc-deployment' inside the worker first."
                ),
            ) from exc
        except Exception as exc:
            logger.exception(f"Failed to trigger pipeline for video '{video.video_id}'")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to submit video '{video.video_id}': {exc}",
            ) from exc

    return UploadResponse(
        run_id=run_id,
        user_id=request.user_id,
        video_count=len(results),
        results=results,
        status="SCHEDULED",
        message=f"{len(results)} video(s) submitted for processing",
        deployment_name=DEPLOY_IDENTIFIER,
    )
