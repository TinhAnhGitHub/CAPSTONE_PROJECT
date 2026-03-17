"""Elasticsearch client for OCR text indexing."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from elasticsearch import AsyncElasticsearch, helpers
from loguru import logger

from video_pipeline.core.client.storage.elasticsearch.config import ElasticsearchSettings
from video_pipeline.core.client.storage.elasticsearch.utils import (
    clean_ocr_text,
    get_ocr_index_mapping,
)


class ElasticsearchOCRClient:
    """
    Elasticsearch client for OCR text indexing.

    Prepares and indexes OCR text documents with:
    - Text cleaning and normalization
    - Optional semantic embeddings via MMBertClient
    - BM25-ready analyzers
    """

    def __init__(self, settings: ElasticsearchSettings, embedding_client=None):
        self.settings = settings
        self._client: AsyncElasticsearch | None = None
        self._embedding_client = embedding_client  # MMBertClient instance

    async def connect(self) -> None:
        """Initialize the Elasticsearch client."""
        if self._client is None:
            self._client = AsyncElasticsearch(**self.settings.get_client_kwargs())
            logger.info(f"[ElasticsearchOCRClient] Connected to {self.settings.url}")

    async def close(self) -> None:
        """Close the Elasticsearch client."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("[ElasticsearchOCRClient] Connection closed")

    async def create_index_if_not_exists(self) -> None:
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        if not await self._client.indices.exists(index=self.settings.index_name):
            mapping = get_ocr_index_mapping(self.settings.embedding_dim)
            await self._client.indices.create(index=self.settings.index_name, body=mapping)
            logger.info(f"[ElasticsearchOCRClient] Created index '{self.settings.index_name}'")
        else:
            logger.info(f"[ElasticsearchOCRClient] Index '{self.settings.index_name}' already exists")

    async def embed_text(self, text: str) -> list[float] | None:
        """Generate embedding for text using MMBertClient."""
        if self._embedding_client is None:
            return None

        embeddings = await self._embedding_client.ainfer([text])
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        return None

    async def embed_texts(self, texts: list[str]) -> list[list[float] | None]:
        """Generate embeddings for multiple texts using MMBertClient."""
        if self._embedding_client is None:
            return [None] * len(texts)

        embeddings = await self._embedding_client.ainfer(texts)
        if embeddings:
            return embeddings
        return [None] * len(texts)

    async def prepare_ocr_document(
        self,
        artifact_id: str,
        raw_text: str,
        user_id: str,
        frame_index: int,
        timestamp: str,
        timestamp_sec: float,
        video_id: str,
        related_video_fps: float,
        image_minio_url: str,
        image_id: str,
        generate_embedding: bool = True,
    ) -> dict[str, Any]:
        """
        Prepare an OCR document for indexing.

        Returns a document dict ready for bulk indexing.
        """
        cleaned_text = clean_ocr_text(raw_text)

        content_vector = None
        if generate_embedding and cleaned_text:
            content_vector = await self.embed_text(cleaned_text)

        return {
            "artifact_id": artifact_id,
            "user_id": user_id,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "frame_index": frame_index,
            "timestamp": timestamp,
            "timestamp_sec": timestamp_sec,
            "video_id": video_id,
            "related_video_fps": related_video_fps,
            "image_minio_url": image_minio_url,
            "image_id": image_id,
            "content_vector": content_vector or [],
            "indexed_at": datetime.now(timezone.utc).isoformat()
        }

    async def index_ocr_document(
        self,
        artifact_id: str,
        raw_text: str,
        user_id: str,
        frame_index: int,
        timestamp: str,
        timestamp_sec: float,
        video_id: str,
        related_video_fps: float,
        image_minio_url: str,
        image_id: str,
        generate_embedding: bool = True,
    ) -> str:
        """
        Index a single OCR document.

        Returns the artifact ID.
        """
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        doc = await self.prepare_ocr_document(
            artifact_id=artifact_id,
            raw_text=raw_text,
            user_id=user_id,
            frame_index=frame_index,
            timestamp=timestamp,
            timestamp_sec=timestamp_sec,
            video_id=video_id,
            related_video_fps=related_video_fps,
            image_minio_url=image_minio_url,
            image_id=image_id,
            generate_embedding=generate_embedding,
        )

        await self._client.index(
            index=self.settings.index_name,
            id=artifact_id,
            document=doc
        )
        logger.debug(f"[ElasticsearchOCRClient] Indexed document {artifact_id}")
        return artifact_id

    async def batch_index_ocr_documents(
        self,
        documents: list[dict[str, Any]],
        generate_embeddings: bool = True,
    ) -> int:
        """
        Bulk index OCR documents.

        Args:
            documents: List of document dicts with fields matching prepare_ocr_document
            generate_embeddings: Whether to generate embeddings for cleaned_text

        Returns number of documents indexed.
        """
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        actions = []
        for doc in documents:
            raw_text = doc.get("raw_text", "")
            cleaned_text = clean_ocr_text(raw_text)

            content_vector = doc.get("content_vector")
            if content_vector is None and generate_embeddings and cleaned_text:
                content_vector = await self.embed_text(cleaned_text)

            action = {
                "_index": self.settings.index_name,
                "_id": doc.get("artifact_id"),
                "_source": {
                    "artifact_id": doc.get("artifact_id"),
                    "user_id": doc.get("user_id"),
                    "raw_text": raw_text,
                    "cleaned_text": cleaned_text,
                    "frame_index": doc.get("frame_index", 0),
                    "timestamp": doc.get("timestamp", ""),
                    "timestamp_sec": doc.get("timestamp_sec", 0.0),
                    "video_id": doc.get("video_id"),
                    "related_video_fps": doc.get("related_video_fps", 0.0),
                    "image_minio_url": doc.get("image_minio_url", ""),
                    "image_id": doc.get("image_id", ""),
                    "content_vector": content_vector or [],
                    "indexed_at": datetime.now(timezone.utc).isoformat()
                }
            }
            actions.append(action)

        success, _ = await helpers.async_bulk(self._client, actions, raise_on_error=True)
        logger.info(f"[ElasticsearchOCRClient] Bulk indexed {success} document(s)")
        return success

    async def delete_by_video_id(self, video_id: str) -> int:
        """Delete all OCR documents for a given video ID."""
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        resp = await self._client.delete_by_query(
            index=self.settings.index_name,
            body={"query": {"term": {"video_id": video_id}}}
        )
        deleted = resp.get("deleted", 0)
        logger.info(f"[ElasticsearchOCRClient] Deleted {deleted} document(s) for video {video_id}")
        return deleted

    async def health_check(self) -> bool:
        """Check Elasticsearch cluster health."""
        if not self._client:
            return False
        try:
            resp = await self._client.cluster.health()
            return resp.get("status") in ("green", "yellow")
        except Exception as e:
            logger.error(f"[ElasticsearchOCRClient] Health check failed: {e}")
            return False