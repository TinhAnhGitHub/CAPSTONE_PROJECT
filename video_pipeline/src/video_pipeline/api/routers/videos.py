"""Router for video deletion endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from video_pipeline.api.services.deletion import VideoDeletionService

router = APIRouter(prefix="/videos", tags=["videos"])


class DeletionResponse(BaseModel):
    """Response for video deletion."""

    video_id: str
    postgres: dict = Field(default_factory=dict)
    minio: dict = Field(default_factory=dict)
    qdrant: dict = Field(default_factory=dict)
    summary: dict = Field(default_factory=dict)
    error: str | None = None


@router.delete(
    "/{video_id}",
    response_model=DeletionResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete all artifacts for a video",
    description=(
        "Delete all artifacts associated with a video_id across all storage backends: "
        "PostgreSQL (artifacts & lineage), MinIO (objects), and Qdrant (vectors)."
    ),
)
async def delete_video(video_id: str) -> DeletionResponse:
    """Delete all artifacts for a given video_id.

    This endpoint will:
    1. Query PostgreSQL for all artifacts with related_video_id or artifact_id == video_id
    2. Delete objects from MinIO using object_name from artifact metadata
    3. Delete lineage records from PostgreSQL
    4. Delete artifacts from PostgreSQL
    5. Delete vectors from Qdrant

    Args:
        video_id: The video ID to delete all artifacts for

    Returns:
        DeletionResponse with results from each storage backend

    Raises:
        HTTPException: If deletion fails with unexpected error
    """
    service = VideoDeletionService()
    result = await service.delete_video(video_id)

    if result.get("error"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deletion failed: {result['error']}",
        )

    return DeletionResponse(**result)