"""Elasticsearch client for OCR text search.

This client provides text search capabilities for OCR-extracted content
from video frames, supporting BM25 keyword search and hybrid search with
semantic embeddings.
"""

from __future__ import annotations

from typing import Any

from elasticsearch import AsyncElasticsearch
from loguru import logger
from pydantic import BaseModel, Field

from videodeepsearch.clients.storage.elasticsearch.schema import ElasticsearchConfig


class OCRSearchResult(BaseModel):
    """Container for a single OCR search result."""

    artifact_id: str
    video_id: str
    user_id: str
    frame_index: int
    timestamp: str
    timestamp_sec: float
    ocr_text: str
    cleaned_text: str
    image_minio_url: str
    image_id: str
    score: float
    highlights: list[str] = Field(default_factory=list)
    related_video_fps: float | None = Field(default=None)

    def brief_representation(self) -> str:
        """Return a brief string representation."""
        text_preview = self.ocr_text[:50] + "..." if len(self.ocr_text) > 50 else self.ocr_text
        return (
            f"score={self.score:.3f} | {self.video_id} "
            f"@ Frame {self.frame_index} ({self.timestamp}) | "
            f"OCR: {text_preview}"
        )

    def detailed_representation(self) -> str:
        """Return a detailed string representation."""
        highlights_str = ""
        if self.highlights:
            highlights_str = "\n  Highlights: " + " ... ".join(self.highlights[:3])

        return (
            f"OCR Result | Video: {self.video_id}\n"
            f"- Frame: {self.frame_index} @ {self.timestamp} ({self.timestamp_sec:.2f}s)\n"
            f"- Score: {self.score:.3f}\n"
            f"- OCR Text: {self.ocr_text[:200]}{'...' if len(self.ocr_text) > 200 else ''}\n"
            f"- Image URL: {self.image_minio_url}{highlights_str}"
        )


class ElasticsearchOCRClient:
    """Elasticsearch client for OCR text search.

    Provides BM25 text search and hybrid search capabilities for
    OCR-extracted text from video frames.

    Features:
    - BM25 keyword search with fuzzy matching
    - kNN semantic vector search
    - RRF (Reciprocal Rank Fusion) hybrid search
    - Text highlighting in results
    """

    def __init__(self, config: ElasticsearchConfig, embedding_client=None):
        """Initialize the Elasticsearch OCR client.

        Args:
            config: Elasticsearch configuration
            embedding_client: Optional MMBertClient for semantic embeddings
        """
        self.config = config
        self._client: AsyncElasticsearch | None = None
        self._embedding_client = embedding_client

    async def connect(self) -> None:
        """Initialize the Elasticsearch client."""
        if self._client is None:
            self._client = AsyncElasticsearch(**self.config.get_client_kwargs())
            logger.info(f"[ElasticsearchOCRClient] Connected to {self.config.url}")

    async def close(self) -> None:
        """Close the Elasticsearch client."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("[ElasticsearchOCRClient] Connection closed")

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

    async def search_text(
        self,
        query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        fuzzy: bool = True,
        highlight: bool = True,
    ) -> list[OCRSearchResult]:
        """Search OCR text using BM25 keyword search.

        Args:
            query: Text query to search for
            top_k: Number of results to return
            video_ids: Optional list of video IDs to filter
            user_id: Optional user ID to filter
            fuzzy: Enable fuzzy matching for better recall
            highlight: Enable text highlighting in results

        Returns:
            List of OCRSearchResult objects
        """
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        # Build filters
        filters = []
        if video_ids:
            filters.append({"terms": {"video_id": video_ids}})
        if user_id:
            filters.append({"term": {"user_id": user_id}})

        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["cleaned_text^1.5", "raw_text"],
                                "type": "best_fields",
                                "fuzziness": "AUTO" if fuzzy else "0",
                                "operator": "or",
                            }
                        }
                    ],
                    "filter": filters,
                }
            },
        }

        if highlight:
            body["highlight"] = {
                "pre_tags": ["<mark>"],
                "post_tags": ["</mark>"],
                "fields": {
                    "cleaned_text": {"fragment_size": 200, "number_of_fragments": 3},
                    "raw_text": {"fragment_size": 200, "number_of_fragments": 1},
                },
            }

        try:
            resp = await self._client.search(index=self.config.index_name, body=body)
            return self._parse_response(resp, highlight)
        except Exception as e:
            logger.error(f"[ElasticsearchOCRClient] Search failed: {e}")
            raise

    async def search_hybrid(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        highlight: bool = True,
    ) -> list[OCRSearchResult]:
        """Search OCR text using hybrid BM25 + kNN with RRF fusion.

        Requires Elasticsearch 8.14+ for RRF support.

        Args:
            query: Text query for BM25 search
            query_vector: Embedding vector for kNN search
            top_k: Number of results to return
            video_ids: Optional list of video IDs to filter
            user_id: Optional user ID to filter
            highlight: Enable text highlighting in results

        Returns:
            List of OCRSearchResult objects
        """
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        # Build filters
        filters = []
        if video_ids:
            filters.append({"terms": {"video_id": video_ids}})
        if user_id:
            filters.append({"term": {"user_id": user_id}})

        filter_clause = {"filter": filters} if filters else {}

        body = {
            "size": top_k,
            "retriever": {
                "rrf": {
                    "window_size": 50,
                    "rank_constant": 60,
                    "retrievers": [
                        # BM25 retriever
                        {
                            "standard": {
                                "query": {
                                    "bool": {
                                        "should": [
                                            {
                                                "multi_match": {
                                                    "query": query,
                                                    "fields": ["cleaned_text^1.5", "raw_text"],
                                                    "type": "best_fields",
                                                    "fuzziness": "AUTO",
                                                    "operator": "or",
                                                }
                                            }
                                        ],
                                        **filter_clause,
                                    }
                                }
                            }
                        },
                        # kNN retriever
                        {
                            "knn": {
                                "field": "content_vector",
                                "query_vector": query_vector,
                                "k": top_k,
                                "num_candidates": 100,
                                **filter_clause,
                            }
                        },
                    ],
                }
            },
        }

        if highlight:
            body["highlight"] = {
                "pre_tags": ["<mark>"],
                "post_tags": ["</mark>"],
                "fields": {
                    "cleaned_text": {"fragment_size": 200, "number_of_fragments": 3},
                },
            }

        try:
            resp = await self._client.search(index=self.config.index_name, body=body)
            return self._parse_response(resp, highlight)
        except Exception as e:
            logger.error(f"[ElasticsearchOCRClient] Hybrid search failed: {e}")
            raise

    async def search_with_embedding(
        self,
        query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        use_hybrid: bool = True,
        highlight: bool = True,
    ) -> list[OCRSearchResult]:
        """Search OCR text with automatic embedding generation.

        If embedding_client is configured and use_hybrid is True,
        performs hybrid search. Otherwise falls back to BM25 only.

        Args:
            query: Text query to search for
            top_k: Number of results to return
            video_ids: Optional list of video IDs to filter
            user_id: Optional user ID to filter
            use_hybrid: Enable hybrid search if embedding client is available
            highlight: Enable text highlighting in results

        Returns:
            List of OCRSearchResult objects
        """
        if use_hybrid and self._embedding_client:
            embeddings = await self._embedding_client.ainfer([query])
            if embeddings:
                return await self.search_hybrid(
                    query=query,
                    query_vector=embeddings[0],
                    top_k=top_k,
                    video_ids=video_ids,
                    user_id=user_id,
                    highlight=highlight,
                )

        # Fallback to BM25 only
        return await self.search_text(
            query=query,
            top_k=top_k,
            video_ids=video_ids,
            user_id=user_id,
            highlight=highlight,
        )

    async def get_ocr_by_video_id(
        self,
        video_id: str,
        user_id: str | None = None,
        limit: int = 1000,
    ) -> list[OCRSearchResult]:
        """Get all OCR documents for a specific video.

        Args:
            video_id: Video ID to filter by
            user_id: Optional user ID to filter by
            limit: Maximum number of results

        Returns:
            List of OCRSearchResult objects
        """
        if not self._client:
            raise RuntimeError("Elasticsearch client not connected")

        filters = [{"term": {"video_id": video_id}}]
        if user_id:
            filters.append({"term": {"user_id": user_id}})

        body = {
            "size": limit,
            "query": {"bool": {"filter": filters}},
            "sort": [{"frame_index": {"order": "asc"}}],
        }

        try:
            resp = await self._client.search(index=self.config.index_name, body=body)
            return self._parse_response(resp, highlight=False)
        except Exception as e:
            logger.error(f"[ElasticsearchOCRClient] Get OCR by video failed: {e}")
            raise

    def _parse_response(
        self, response: dict, highlight: bool = True
    ) -> list[OCRSearchResult]:
        """Parse Elasticsearch response into OCRSearchResult objects."""
        results = []

        for hit in response.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})

            highlights = []
            if highlight and "highlight" in hit:
                highlights = hit["highlight"].get("cleaned_text", [])

            result = OCRSearchResult(
                artifact_id=src.get("artifact_id", ""),
                video_id=src.get("video_id", ""),
                user_id=src.get("user_id", ""),
                frame_index=src.get("frame_index", 0),
                timestamp=src.get("timestamp", ""),
                timestamp_sec=src.get("timestamp_sec", 0.0),
                ocr_text=src.get("raw_text", ""),
                cleaned_text=src.get("cleaned_text", ""),
                image_minio_url=src.get("image_minio_url", ""),
                image_id=src.get("image_id", ""),
                score=hit.get("_score", 0.0),
                highlights=highlights,
                related_video_fps=src.get("related_video_fps"),
            )
            results.append(result)

        return results


__all__ = ["ElasticsearchOCRClient", "OCRSearchResult", "ElasticsearchConfig"]