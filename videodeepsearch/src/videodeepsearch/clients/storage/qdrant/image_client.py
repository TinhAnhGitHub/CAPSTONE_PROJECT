"""Image Qdrant client for searching image embeddings."""

from __future__ import annotations

from qdrant_client.models import ScoredPoint

from videodeepsearch.clients.storage.qdrant.client import BaseQdrantClient
from videodeepsearch.schemas import ImageInterface

IMAGE_DENSE_FIELD = "image_dense"


class ImageQdrantClient(BaseQdrantClient[ImageInterface]):
    """Client for searching image embeddings in Qdrant.

    The image collection contains:
    - image_dense: Dense embedding from visual encoder (e.g., CLIP)
    - Payload: frame_index, timestamp, timestamp_sec, related_video_id,
               related_video_fps, image_minio_url, user_id, minio_url, caption_text
    """

    @staticmethod
    def _hit_to_item(hit: ScoredPoint) -> ImageInterface:
        """Convert a Qdrant hit to ImageInterface.

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
                minio_path=str(payload.get("image_minio_url", payload.get("minio_url", ""))),
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

    async def search_visual(
        self,
        query_vector: list[float],
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[ImageInterface]:
        """Search images by visual similarity.

        Args:
            query_vector: Visual embedding of the query
            video_ids: Optional list of video IDs to filter by
            user_id: Optional user ID to filter by
            limit: Maximum number of results

        Returns:
            List of matching images
        """
        query_filter = self.build_filter(video_ids=video_ids, user_id=user_id)
        return await self.search_dense(
            query_vector=query_vector,
            vector_name=IMAGE_DENSE_FIELD,
            limit=limit,
            query_filter=query_filter,
        )