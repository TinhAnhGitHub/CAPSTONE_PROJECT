from __future__ import annotations
from qdrant_client.models import ScoredPoint
from videodeepsearch.clients.storage.qdrant.client import BaseQdrantClient
from videodeepsearch.schemas import ImageInterface

IMAGE_DENSE_FIELD = "image_dense"
CAPTION_TEXT_DENSE_FIELD = "image_caption_dense"
CAPTION_SPARSE_FIELD = "image_caption_sparse"

class ImageQdrantClient(BaseQdrantClient[ImageInterface]):
    """Client for searching image embeddings in Qdrant.

    The image collection contains:
    - image_dense: Dense embedding from visual encoder (e.g., CLIP)
    - Payload: frame_index, timestamp, timestamp_sec, related_video_id,
               related_video_fps, image_minio_url, user_id, minio_url, caption_text
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
        self.image_collection_name = self.base_collection_name + "_image"
        self.image_caption_collection_name = self.base_collection_name + "_image_caption"

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

    async def search_image_dense_qwenvl(
        self,
        query_vector: list[float],
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[ImageInterface]:
        """Search images by using qwenvl embeddings.

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
            collection_name=self.image_collection_name
        )
    
    async def search_image_dense_mmbert(
        self,
        query_vector: list[float],
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[ImageInterface]:
        """Search images by using mmbert embeddings.

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
            vector_name=CAPTION_TEXT_DENSE_FIELD,
            limit=limit,
            query_filter=query_filter,
            collection_name=self.image_caption_collection_name
    )
        
    async def search_image_hybrid_mmbert(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[ImageInterface]:
        """Search images by visual similarity and metadata using mmbert embeddings.

         Args:
            query_vector: Visual embedding of the query
            video_ids: Optional list of video IDs to filter by
            user_id: Optional user ID to filter by
            limit: Maximum number of results
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
            collection_name=self.image_caption_collection_name,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )
        
        
        
        
    
