"""Router for video endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field
from typing import Any

from video_pipeline.api.services.deletion import VideoDeletionService
from video_pipeline.api.services.retrieval import VideoRetrievalService

router = APIRouter(prefix="/videos", tags=["videos"])


class DeletionResponse(BaseModel):
    """Response for video deletion."""

    video_id: str
    postgres: dict[str, Any] = Field(default_factory=dict)
    minio: dict[str, Any] = Field(default_factory=dict)
    qdrant: dict[str, Any] = Field(default_factory=dict)
    arango: dict[str, Any] = Field(default_factory=dict)
    elasticsearch: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class RetrievalResponse(BaseModel):
    """Response for video data retrieval."""

    video_id: str
    postgres: dict[str, Any] = Field(default_factory=dict)
    arango: dict[str, Any] = Field(default_factory=dict)
    qdrant: dict[str, Any] = Field(default_factory=dict)
    elasticsearch: dict[str, Any] = Field(default_factory=dict)


@router.delete(
    "/{video_id}",
    response_model=DeletionResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete all artifacts for a video",
    description=(
        "Delete all artifacts associated with a video_id across all storage backends: "
        "PostgreSQL (artifacts & lineage), MinIO (objects), Qdrant (vectors), "
        "ArangoDB (knowledge graph), and Elasticsearch (OCR text)."
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
    6. Delete KG data from ArangoDB
    7. Delete OCR documents from Elasticsearch

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


@router.get(
    "/{video_id}/full",
    response_model=RetrievalResponse,
    summary="Get all data for a video",
    description=(
        "Retrieve all data associated with a video_id from all storage backends: "
        "PostgreSQL (artifacts & lineage), ArangoDB (KG), Qdrant (embeddings), "
        "and Elasticsearch (OCR text)."
    ),
)
async def get_video_data(
    video_id: str,
    sources: str | None = Query(
        default=None,
        description="Comma-separated list of sources: postgres,arango,qdrant,elasticsearch",
    ),
    include_vectors: bool = Query(
        default=False,
        description="Include embedding vectors in Qdrant results (large response)",
    ),
) -> RetrievalResponse:
    """Retrieve all data associated with a video_id.

    This endpoint fetches data from multiple storage backends in parallel:
    - PostgreSQL: Artifacts metadata and lineage
    - ArangoDB: Knowledge graph (entities, events, communities, relationships)
    - Qdrant: Embedding vectors (images, captions, segments)
    - Elasticsearch: OCR text documents

    Args:
        video_id: The video ID to retrieve data for
        sources: Optional comma-separated list of sources. Fetches all if not provided.
        include_vectors: Whether to include embedding vectors (increases response size)

    Returns:
        RetrievalResponse with data from each requested source
    """
    source_list = None
    if sources:
        source_list = [s.strip() for s in sources.split(",") if s.strip()]

    service = VideoRetrievalService()
    result = await service.get_video_data(
        video_id=video_id,
        sources=source_list,
        include_vectors=include_vectors,
    )

    return RetrievalResponse(
        video_id=result.get("video_id", video_id),
        postgres=result.get("postgres", {}),
        arango=result.get("arango", {}),
        qdrant=result.get("qdrant", {}),
        elasticsearch=result.get("elasticsearch", {}),
    )