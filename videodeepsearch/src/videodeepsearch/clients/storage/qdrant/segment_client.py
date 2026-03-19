"""Segment Qdrant client for searching segment embeddings."""

from __future__ import annotations

from qdrant_client.models import ScoredPoint

from videodeepsearch.clients.storage.qdrant.client import BaseQdrantClient
from videodeepsearch.schemas import SegmentInterface
from videodeepsearch.schemas.artifacts import ImageInterface


SEGMENT_DENSE_FIELD = "segment_dense"
SEGMENT_CAPTION_DENSE_FIELD = "segment_caption_dense"
SEGMENT_CAPTION_SPARSE_FIELD = "segment_caption_sparse"

class SegmentQdrantClient(BaseQdrantClient[SegmentInterface]):
    """Client for searching segment embeddings in Qdrant.

    The segment collection contains:
    - segment_dense: Dense embedding from segment encoder
    - Payload: related_audio_segment_artifact_id, related_video_id,
               start_frame, end_frame, start_timestamp, end_timestamp,
               start_sec, end_sec, frame_indices, embedding_dim, user_id,
               minio_url, caption_text
    """

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        grpc_port: int = 6334,
        prefer_grpc: bool = True,
    ):
        super().__init__(
            host=host,
            port=port,
            collection_name=collection_name,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
        )
        self.segment_collection_name = self.base_collection_name + "_segment"
        self.segment_caption_collection_name = self.base_collection_name + "_segment_caption"

    @staticmethod
    def _hit_to_item(hit: ScoredPoint) -> SegmentInterface:
        """Convert a Qdrant hit to SegmentInterface.

        Args:
            hit: Qdrant search result

        Returns:
            SegmentInterface object

        Raises:
            KeyError: If required fields are missing from the payload
        """
        payload = hit.payload or {}

        try:
            start_sec = payload.get("start_sec")
            if start_sec is not None:
                start_sec = float(start_sec)

            end_sec = payload.get("end_sec")
            if end_sec is not None:
                end_sec = float(end_sec)

            frame_indices = payload.get("frame_indices")
            if frame_indices is not None:
                frame_indices = [int(idx) for idx in frame_indices]

            return SegmentInterface(
                id=str(payload.get("id", hit.id)),
                related_video_id=str(payload["related_video_id"]),
                minio_path=str(payload.get("minio_url", "")),
                user_bucket=str(payload["user_id"]),
                start_frame=int(payload["start_frame"]),
                end_frame=int(payload["end_frame"]),
                start_time=str(payload["start_timestamp"]),
                end_time=str(payload["end_timestamp"]),
                segment_caption=str(payload.get("caption_text", "")),
                score=float(hit.score),
                start_sec=start_sec,
                end_sec=end_sec,
                frame_indices=frame_indices,
            )
        except KeyError as e:
            raise KeyError(f"Missing expected field in Qdrant payload: {e}") from e

    async def search_segment_dense_qwenvl(
        self,
        query_vector: list[float],
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[SegmentInterface]:
        """Search segments by semantic similarity.

        Args:
            query_vector: Segment embedding query
            video_ids: Optional list of video IDs to filter by
            user_id: Optional user ID to filter by
            limit: Maximum number of results

        Returns:
            List of matching segments
        """
        query_filter = self.build_filter(video_ids=video_ids, user_id=user_id)
        return await self.search_dense(
            query_vector=query_vector,
            vector_name=SEGMENT_DENSE_FIELD,
            limit=limit,
            query_filter=query_filter,
            collection_name=self.segment_collection_name
        )
    
    async def search_segment_dense_mmbert(
        self,
        query_vector: list[float],
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[SegmentInterface]:
        """Search segments by semantic similarity using mmbert embeddings.

        Args:
            query_vector: Segment embedding query
            video_ids: Optional list of video IDs to filter by
            user_id: Optional user ID to filter by
            limit: Maximum number of results

        Returns:
            List of matching segments
        """
        query_filter = self.build_filter(video_ids=video_ids, user_id=user_id)
        return await self.search_dense(
            query_vector=query_vector,
            vector_name=SEGMENT_CAPTION_DENSE_FIELD,
            limit=limit,
            query_filter=query_filter,
            collection_name=self.segment_caption_collection_name
        )
    
    async def search_segment_hybrid_mmbert(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[SegmentInterface]:
        """Search segments using hybrid mmbert embeddings.

        Args:
            dense_vector: Dense segment embedding query
            sparse_vector: Sparse segment embedding query
            video_ids: Optional list of video IDs to filter by
            user_id: Optional user ID to filter by
            limit: Maximum number of results
            sparse_weight: Weight for the sparse vector in hybrid scoring

        Returns:
            List of matching segments
        """
        query_filter = self.build_filter(video_ids=video_ids, user_id=user_id)
        return await self.search_hybrid(
            dense_vector=dense_vector,
            dense_vector_name=SEGMENT_CAPTION_DENSE_FIELD,
            sparse_vector=sparse_vector,
            sparse_vector_name=SEGMENT_CAPTION_SPARSE_FIELD,
            limit=limit,
            query_filter=query_filter,
            collection_name=self.segment_caption_collection_name,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )