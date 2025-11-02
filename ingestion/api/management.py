from fastapi import HTTPException, APIRouter, Depends, status, Query
from pydantic import BaseModel, Field
from core.management.cleanup import ArtifactDeleter, DeletionResult
from core.management.status import VideoStatusInfo
from core.dependencies.application import (
    get_artifact_deleter,
    get_video_status_manager
)
from core.config.logging import run_logger

from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterName
from prefect.client.schemas.sorting import FlowRunSort
from prefect.states import Cancelled
from prefect.server.schemas.states import StateType





router = APIRouter(prefix='/management', tags=["management"])


class DeletionResponse(BaseModel):
    """Response model for deletion operations."""
    success: bool
    video_id: str
    metadata:dict


class MilvusDeletionResponse(BaseModel):
    success: bool
    user_id: str
    related_video_id: str
    total_deleted: int
    per_collection_deleted: dict[str, int]
    errors: list[str] = Field(default_factory=list)


class RunCancellationResponse(BaseModel):
    run_id: str
    flow_run_id: str
    flow_run_state: str | None
    cancellation_request: bool
    video_deletions: list[DeletionResponse]

TERMINAL_FLOW_STATES = {
    StateType.COMPLETED,
    StateType.CANCELLED,
    StateType.CRASHED,
    StateType.FAILED,
}




@router.delete(
    "/videos/{video_id}",
    response_model=DeletionResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete video and all derived artifacts",
    description="Cascading deletion of video and all derived artifacts (segments, transcripts, embeddings, etc.)"
)
async def delete_video(
    video_id: str,
    deleter: ArtifactDeleter = Depends(get_artifact_deleter)
) -> DeletionResponse:
    result: DeletionResult = await deleter.delete_video_cascade(video_id)
    return DeletionResponse(
        success=result.success,
        video_id=result.video_id,
        metadata=result.metadata
    )


@router.delete(
    "/videos/{video_id}/stages/{artifact_type}",
    response_model=DeletionResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete artifacts from a specific stage",
    description="Delete all artifacts of a specific type/stage for a video, including descendants"
)
async def delete_video_stage(
    video_id: str,
    artifact_type: str,
    deleter: ArtifactDeleter = Depends(get_artifact_deleter)
) -> DeletionResponse:
    """Delete all artifacts of a specific stage/type for a video."""
    try:
        result: DeletionResult = await deleter.delete_stage_artifacts(
            video_id=video_id,
            artifact_type=artifact_type
        )
        return DeletionResponse(
            success=result.success,
            video_id=result.video_id,
            metadata=result.metadata
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete stage artifacts: {str(e)}"
        )

@router.get(
    "/videos/{video_id}/status",
    response_model=VideoStatusInfo,
    status_code=status.HTTP_200_OK,
    summary="Get video processing status",
    description="Retrieve detailed processing status for a video by ID or name"
)
async def get_video_status(
    video_id: str,
    status_manager=Depends(get_video_status_manager)
) -> VideoStatusInfo:
    """Get the current processing status of a video."""
    
    try:
        video_status = await status_manager.get_video_status(video_id)
        if video_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video '{video_id}' not found"
            )
        return video_status
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve status: {str(e)}"
        )
    



@router.post(
    '/runs/{run_id}/cancel',
    response_model=RunCancellationResponse,
    status_code=status.HTTP_200_OK,
    summary="Cancel Prefect run and remove related artifacts",
    description="Cancels the Prefect flow run identified by run_id and deletes associated video artifacts.",
)
async def cancel_run(
    run_id: str,
    deleter: ArtifactDeleter=Depends(get_artifact_deleter)
):
    async with get_client() as client:
        flow_runs = await client.read_flow_runs(
            flow_run_filter=FlowRunFilter(name=FlowRunFilterName(any_=[run_id])), sort=FlowRunSort.START_TIME_DESC, limit=1
        )
        if not flow_runs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No Prefect flow run found for run_id '{run_id}'",
            )

        flow_run = flow_runs[0]
        flow_run_id = str(flow_run.flow_id)

        cancellation_requested = False
        state_type = flow_run.state.type if flow_run.state else None
        if state_type not in TERMINAL_FLOW_STATES:
            await client.set_flow_run_state(
                flow_run.id,
                state=Cancelled(message="Cancelled via management API"),
            )
            cancellation_requested = True
            flow_run = await client.read_flow_run(flow_run.id)

        parameters = flow_run.parameters or {}

    video_ids: list[str] = []
    for entry in parameters.get('video_files') or []:
        video_ids.append(entry[0])


    video_deletions: list[DeletionResponse] = []
    for video_id in video_ids:
        try:
            result = await deleter.delete_video_cascade(video_id)
            video_deletions.append(
                DeletionResponse(
                    success=result.success,
                    video_id=result.video_id,
                    metadata=result.metadata,
                )
            )
        except Exception as exc:
            run_logger.exception(
                "Failed to delete video artifacts after cancellation (run_id=%s, video_id=%s)",
                run_id,
                video_id,
            )
            video_deletions.append(
                DeletionResponse(
                    success=False,
                    video_id=video_id,
                    metadata={"error": str(exc)},
                )
            )
    return RunCancellationResponse(
        run_id=run_id,
        flow_run_id=flow_run_id,
        flow_run_state=flow_run.state.name if flow_run.state else "UNKNOWN", 
        cancellation_request=cancellation_requested,
        video_deletions=video_deletions,
    )
