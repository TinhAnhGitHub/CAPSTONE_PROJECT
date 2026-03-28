"""Audio Qdrant client for searching audio transcript embeddings."""

from __future__ import annotations

from qdrant_client.models import ScoredPoint

from videodeepsearch.clients.storage.qdrant.client import BaseQdrantClient
from videodeepsearch.schemas import AudioInterface


AUDIO_TRANSCRIPT_DENSE_FIELD = "audio_transcript_dense"
AUDIO_TRANSCRIPT_SPARSE_FIELD = "audio_transcript_sparse"


class AudioQdrantClient(BaseQdrantClient[AudioInterface]):
    """Client for searching audio transcript embeddings in Qdrant.

    The audio transcript collection contains:
    - audio_transcript_dense: Dense embedding from mmBERT (768-dim)
    - audio_transcript_sparse: Sparse embedding from SPLADE
    - Payload: related_audio_segment_artifact_id, related_video_id,
               segment_index, start_frame, end_frame, start_timestamp,
               end_timestamp, start_sec, end_sec, embedding_dim,
               user_id, minio_url, audio_text
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
        self.audio_collection_name = self.base_collection_name + "_audio_transcript"

    @staticmethod
    def _hit_to_item(hit: ScoredPoint) -> AudioInterface:
        """Convert a Qdrant hit to AudioInterface.

        Args:
            hit: Qdrant search result

        Returns:
            AudioInterface object

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

            return AudioInterface(
                id=str(payload.get("id", hit.id)),
                related_video_id=str(payload["related_video_id"]),
                minio_path=str(payload.get("minio_url", "")),
                user_bucket=str(payload["user_id"]),
                segment_index=int(payload["segment_index"]),
                start_frame=int(payload["start_frame"]),
                end_frame=int(payload["end_frame"]),
                start_time=str(payload["start_timestamp"]),
                end_time=str(payload["end_timestamp"]),
                audio_text=str(payload.get("audio_text", "")),
                score=float(hit.score),
                start_sec=start_sec,
                end_sec=end_sec,
                related_audio_segment_artifact_id=payload.get("related_audio_segment_artifact_id"),
            )
        except KeyError as e:
            raise KeyError(f"Missing expected field in Qdrant payload: {e}") from e

    async def search_audio_dense(
        self,
        query_vector: list[float],
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 10,
    ) -> list[AudioInterface]:
        """Search audio transcripts by dense vector similarity.

        Args:
            query_vector: Dense embedding of the query (mmBERT 768-dim)
            video_ids: Optional list of video IDs to filter by
            user_id: Optional user ID to filter by
            limit: Maximum number of results

        Returns:
            List of matching audio segments
        """
        query_filter = self.build_filter(video_ids=video_ids, user_id=user_id)
        return await self.search_dense(
            query_vector=query_vector,
            vector_name=AUDIO_TRANSCRIPT_DENSE_FIELD,
            limit=limit,
            query_filter=query_filter,
            collection_name=self.audio_collection_name,
        )

    async def search_audio_hybrid(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        limit: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[AudioInterface]:
        """Search audio transcripts using hybrid dense + sparse search.

        Uses RRF fusion to combine dense (mmBERT) and sparse (SPLADE) embeddings.

        Args:
            dense_vector: Dense embedding of the query (mmBERT 768-dim)
            sparse_vector: Sparse embedding of the query (SPLADE)
            video_ids: Optional list of video IDs to filter by
            user_id: Optional user ID to filter by
            limit: Maximum number of results
            dense_weight: Weight for dense search results (default 0.7)
            sparse_weight: Weight for sparse search results (default 0.3)

        Returns:
            List of matching audio segments
        """
        query_filter = self.build_filter(video_ids=video_ids, user_id=user_id)
        return await self.search_hybrid(
            dense_vector=dense_vector,
            dense_vector_name=AUDIO_TRANSCRIPT_DENSE_FIELD,
            sparse_vector=sparse_vector,
            sparse_vector_name=AUDIO_TRANSCRIPT_SPARSE_FIELD,
            limit=limit,
            query_filter=query_filter,
            collection_name=self.audio_collection_name,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )