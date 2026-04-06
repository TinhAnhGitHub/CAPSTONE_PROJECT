"""Caption Qdrant client for searching caption embeddings with hybrid search."""

from __future__ import annotations

from qdrant_client.models import ScoredPoint

from videodeepsearch.clients.storage.qdrant.client import BaseQdrantClient
from videodeepsearch.schemas import ImageInterface

CAPTION_TEXT_DENSE_FIELD = "caption_text_dense"
CAPTION_MM_DENSE_FIELD = "caption_mm_dense"
CAPTION_SPARSE_FIELD = "caption_sparse"


class CaptionQdrantClient(BaseQdrantClient[ImageInterface]):
    """Client for searching caption embeddings in Qdrant with hybrid search.

    The caption collection contains:
    - caption_text_dense: Dense mmBERT text embedding (Vietnamese text)
    - caption_mm_dense: Dense QwenVL multimodal embedding
    - caption_sparse: Sparse SPLADE encoding for keyword matching
    - Payload: frame_index, timestamp, timestamp_sec, related_video_id,
               related_video_fps, caption_minio_url, image_minio_url, user_id,
               caption_id, image_id, mm_embedding_minio_url, caption_text
    """

    @staticmethod
    def _hit_to_item(hit: ScoredPoint) -> ImageInterface:
        """Convert a Qdrant hit to ImageInterface.

        Caption search returns images with matching captions.

        Args:
            hit: Qdrant search result

        Returns:
            ImageInterface object

        Raises:
            KeyError: If required fields are missing from the payload
        """
        payload = hit.payload or {}

        try:
            timestamp_sec = payload.get("timestamp_sec")
            if timestamp_sec is not None:
                timestamp_sec = float(timestamp_sec)

            related_video_fps = payload.get("related_video_fps")
            if related_video_fps is not None:
                related_video_fps = float(related_video_fps)

            return ImageInterface(
                id=str(payload.get("id", hit.id)),
                related_video_id=str(payload["related_video_id"]),
                minio_path=str(payload.get("image_minio_url", "")),
                user_bucket=str(payload["user_id"]),
                frame_index=int(payload["frame_index"]),
                timestamp=str(payload["timestamp"]),
                image_caption=str(payload.get("caption_text", "")),
                score=float(hit.score),
                timestamp_sec=timestamp_sec,
                related_video_fps=related_video_fps,
            )
        except KeyError as e:
            raise KeyError(f"Missing expected field in Qdrant payload: {e}") from e

    async def search_text(
        self,
        query_vector: list[float],
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[ImageInterface]:
        """Search images by text caption embedding (mmBERT).

        Args:
            query_vector: Text embedding from mmBERT encoder
            video_ids: Optional list of video IDs to filter by
            user_id: Optional user ID to filter by
            limit: Maximum number of results

        Returns:
            List of matching images
        """
        query_filter = self.build_filter(video_ids=video_ids, user_id=user_id)
        return await self.search_dense(
            query_vector=query_vector,
            vector_name=CAPTION_TEXT_DENSE_FIELD,
            limit=limit,
            query_filter=query_filter,
        )

    async def search_multimodal(
        self,
        query_vector: list[float],
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[ImageInterface]:
        """Search images by multimodal embedding (QwenVL).

        Args:
            query_vector: Multimodal embedding from QwenVL encoder
            video_ids: Optional list of video IDs to filter by
            user_id: Optional user ID to filter by
            limit: Maximum number of results

        Returns:
            List of matching images
        """
        query_filter = self.build_filter(video_ids=video_ids, user_id=user_id)
        return await self.search_dense(
            query_vector=query_vector,
            vector_name=CAPTION_MM_DENSE_FIELD,
            limit=limit,
            query_filter=query_filter,
        )

    async def search_text_sparse(
        self,
        sparse_vector: dict[int, float],
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[ImageInterface]:
        """Search images by sparse SPLADE embedding.

        Args:
            sparse_vector: Sparse vector from SPLADE encoder
            video_ids: Optional list of video IDs to filter by
            user_id: Optional user ID to filter by
            limit: Maximum number of results

        Returns:
            List of matching images
        """
        query_filter = self.build_filter(video_ids=video_ids, user_id=user_id)
        return await super().search_sparse(
            sparse_vector=sparse_vector,
            vector_name=CAPTION_SPARSE_FIELD,
            limit=limit,
            query_filter=query_filter,
        )

    async def search_hybrid_text(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[ImageInterface]:
        """Hybrid search combining text dense and SPLADE sparse vectors.

        Args:
            dense_vector: Dense text embedding from mmBERT
            sparse_vector: Sparse SPLADE embedding
            video_ids: Optional list of video IDs to filter by
            user_id: Optional user ID to filter by
            limit: Maximum number of results
            dense_weight: Weight for dense search (default 0.7)
            sparse_weight: Weight for sparse search (default 0.3)

        Returns:
            List of matching images
        """
        query_filter = self.build_filter(video_ids=video_ids, user_id=user_id)
        return await self.search_hybrid(
            dense_vector=dense_vector,
            dense_vector_name=CAPTION_TEXT_DENSE_FIELD,
            sparse_vector=sparse_vector,
            sparse_vector_name=CAPTION_SPARSE_FIELD,
            limit=limit,
            query_filter=query_filter,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )