"""OCR Search Toolkit for text retrieval from video frames.

This toolkit provides text search capabilities for OCR-extracted content
from video frames, supporting both keyword (BM25) and semantic search.

All tools return ToolResult for unified interface.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from agno.tools import Toolkit, tool
from agno.tools.function import ToolResult
from loguru import logger

from videodeepsearch.clients.storage.elasticsearch import (
    ElasticsearchOCRClient,
    OCRSearchResult,
)
from videodeepsearch.clients.inference import MMBertClient
from videodeepsearch.toolkit.common import CacheManager


class OCRSearchToolkit(Toolkit):
    """Toolkit for OCR text search with dependency injection.

    Provides tools for:
    - Keyword search using BM25
    - Semantic search using MMBert embeddings
    - Hybrid search combining BM25 + kNN with RRF fusion
    - Retrieving all OCR text for a specific video

    All tools support caching and return ToolResult for unified interface.
    """

    def __init__(
        self,
        es_ocr_client: ElasticsearchOCRClient,
        mmbert_client: MMBertClient | None = None,
        cache_ttl: int = 1800,
        cache_dir: str | None = None,
    ):
        """Initialize the OCRSearchToolkit.

        Args:
            es_ocr_client: Elasticsearch OCR client for text search
            mmbert_client: Optional MMBert client for semantic embeddings
            cache_ttl: Cache time-to-live in seconds (default 30 minutes)
            cache_dir: Optional custom cache directory
        """
        self.es_client = es_ocr_client
        self.mmbert = mmbert_client
        self.cache_ttl = cache_ttl
        self.cache_manager = CacheManager(cache_dir)
        self._result_store: dict[str, list[OCRSearchResult]] = {}

        super().__init__(name="OCR Search Tools")

    def _store_result(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        results: list[OCRSearchResult],
    ) -> ToolResult:
        """Store results in memory and return ToolResult.

        Args:
            tool_name: Name of the tool
            kwargs: Tool arguments
            results: Search results

        Returns:
            ToolResult with search results
        """
        handle_id = hashlib.md5(
            json.dumps({"tool": tool_name, "args": kwargs}, sort_keys=True).encode()
        ).hexdigest()[:8]

        self._result_store[handle_id] = results

        content = (
            f"Found {len(results)} OCR result(s).\n"
            f"Handle ID: {handle_id}\n"
            f"To view details: use `view_ocr_result(handle_id='{handle_id}', view_mode='detailed')`\n\n"
            f"{self._get_brief(results, 5)}"
        )

        return ToolResult(content=content)

    def _get_brief(self, results: list[OCRSearchResult], top_n: int = 5) -> str:
        """Get brief representation of top N results."""
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)[:top_n]

        lines = [
            f"Total: {len(results)} OCR result(s)",
            f"Top {min(top_n, len(sorted_results))}:",
        ]

        for i, item in enumerate(sorted_results):
            lines.append(f"  {i}. {item.brief_representation()}")

        return "\n".join(lines)

    def _get_detailed(self, results: list[OCRSearchResult], top_n: int = 5) -> str:
        """Get detailed representation of top N results."""
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)[:top_n]

        lines = [
            "=== Detailed OCR Results ===",
            f"Total: {len(results)} result(s)",
            f"Top {min(top_n, len(sorted_results))}:",
            "",
        ]

        for i, item in enumerate(sorted_results):
            lines.append(f"[{i}] {item.detailed_representation()}")
            lines.append("")

        return "\n".join(lines)

    # @tool(
    #     description=(
    #         "Search for text in video frames using keyword matching (BM25). "
    #         "Best for finding specific words, phrases, or exact text matches. "
    #         "Supports fuzzy matching for handling OCR errors."
    #     ),
    #     instructions=(
    #         "Use when looking for specific words, numbers, or phrases in video frames. "
    #         "Supports fuzzy matching to handle OCR recognition errors. "
    #         "Best for: finding specific text, numbers, labels, captions in videos."
    #     ),
    #     cache_results=True,
    #     cache_ttl=1800,
    # )
    async def search_ocr_text(
        self,
        query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        fuzzy: bool = True,
    ) -> ToolResult:
        """Search OCR text using BM25 keyword matching.

        Best for finding specific words, phrases, or exact text matches.
        Supports fuzzy matching to handle OCR recognition errors.

        Args:
            query: Text to search for (keywords or phrases)
            top_k: Number of results to return (default 10)
            video_ids: Optional list of video IDs to filter
            user_id: Optional user ID to filter
            fuzzy: Enable fuzzy matching for OCR error tolerance (default True)

        Returns:
            ToolResult with OCR search results
        """
        kwargs = {
            "query": query,
            "top_k": top_k,
            "video_ids": video_ids,
            "user_id": user_id,
            "fuzzy": fuzzy,
        }

        try:
            results = await self.es_client.search_text(
                query=query,
                top_k=top_k,
                video_ids=video_ids,
                user_id=user_id,
                fuzzy=fuzzy,
                highlight=True,
            )
            return self._store_result("search_ocr_text", kwargs, results)
        except Exception as e:
            logger.error(f"[OCRSearchToolkit] search_ocr_text failed: {e}")
            return ToolResult(content=f"Error: OCR text search failed - {str(e)}")

    # @tool(
    #     description=(
    #         "Get all OCR text extracted from a specific video. "
    #         "Returns all text found in video frames, sorted by frame order."
    #     ),
    #     instructions=(
    #         "Use when you need all text from a specific video. "
    #         "Useful for reviewing all OCR content or finding text within a known video."
    #     ),
    #     cache_results=True,
    #     cache_ttl=1800,
    # )
    async def get_ocr_by_video(
        self,
        video_id: str,
        user_id: str | None = None,
        limit: int = 1000,
    ) -> ToolResult:
        """Get all OCR text for a specific video.

        Args:
            video_id: Video ID to retrieve OCR for
            user_id: Optional user ID to filter by
            limit: Maximum number of results (default 1000)

        Returns:
            ToolResult with all OCR results for the video
        """
        kwargs = {
            "video_id": video_id,
            "user_id": user_id,
            "limit": limit,
        }

        try:
            results = await self.es_client.get_ocr_by_video_id(
                video_id=video_id,
                user_id=user_id,
                limit=limit,
            )
            return self._store_result("get_ocr_by_video", kwargs, results)
        except Exception as e:
            logger.error(f"[OCRSearchToolkit] get_ocr_by_video failed: {e}")
            return ToolResult(content=f"Error: Failed to get OCR for video - {str(e)}")
        r
    # @tool(
    #     description="View cached OCR search results with different view modes.",
    #     cache_results=False,
    # )
    def view_ocr_result(
        self,
        handle_id: str,
        view_mode: Literal["brief", "detailed", "full"] = "brief",
        top_n: int = 5,
    ) -> ToolResult:
        """View cached OCR search results.

        Args:
            handle_id: Handle ID from previous search
            view_mode: 'brief' (top N), 'detailed' (top N detailed), 'full' (all)
            top_n: Number of results to show in brief/detailed mode

        Returns:
            ToolResult with formatted view of cached results
        """
        if handle_id not in self._result_store:
            return ToolResult(
                content=f"No cached results found for handle_id '{handle_id}'. "
                f"Run a search tool first to generate results."
            )

        results = self._result_store[handle_id]

        if view_mode == "brief":
            return ToolResult(content=self._get_brief(results, top_n))
        elif view_mode == "detailed":
            return ToolResult(content=self._get_detailed(results, top_n))
        else:
            return ToolResult(content=self._get_detailed(results, len(results)))


__all__ = ["OCRSearchToolkit"]