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

    Supports context binding for user_id and video_ids.

    All tools support caching and return ToolResult for unified interface.
    """

    def __init__(
        self,
        es_ocr_client: ElasticsearchOCRClient,
        mmbert_client: MMBertClient | None = None,
        user_id: str | None = None,
        video_ids: list[str] | None = None,
        cache_ttl: int = 1800,
        cache_dir: str | None = None,
    ):
        """Initialize the OCRSearchToolkit.

        Args:
            es_ocr_client: Elasticsearch OCR client for text search
            mmbert_client: Optional MMBert client for semantic embeddings
            user_id: Default user ID for all searches (bound at creation)
            video_ids: Default video IDs for all searches (bound at creation)
            cache_ttl: Cache time-to-live in seconds (default 30 minutes)
            cache_dir: Optional custom cache directory
        """
        self.es_client = es_ocr_client
        self.mmbert = mmbert_client
        self.cache_ttl = cache_ttl
        self.cache_manager = CacheManager(cache_dir)
        self._result_store: dict[str, list[OCRSearchResult]] = {}

        # Context binding
        self._user_id = user_id
        self._video_ids = video_ids

        super().__init__(
            name="OCR Search Tools",
            tools=[
                self.search_ocr_text,
                self.get_ocr_by_video,
                self.view_ocr_result,
            ],
        )

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

    @tool(
        description=(
            "Search for text in video frames using keyword matching (BM25). "
            "Best for finding specific words, phrases, or exact text matches. "
            "Supports fuzzy matching for handling OCR errors.\n\n"
            "Typical workflow - OCR search and verification:\n"
            "  1. This tool - find frames containing specific text\n"
            "  2. view_ocr_result - inspect detailed results with handle_id\n"
            "  3. utility.extract_frames_by_time_window - extract frames for visual verification\n"
            "  4. utility.get_related_asr_from_image - get spoken context around found text\n\n"
            "When to use:\n"
            "  - Looking for specific words, numbers, or phrases in video frames\n"
            "  - Searching for labels, captions, signs, or on-screen text\n"
            "  - Finding documents, slides, or text-heavy content\n\n"
            "Related tools:\n"
            "  - get_ocr_by_video: Get all OCR text for a specific video\n"
            "  - view_ocr_result: Inspect cached search results\n"
            "  - utility.extract_frames_by_time_window: Extract frames at found timestamps\n"
            "  - search.get_images_from_caption_query_mmbert: For image content search (not text)\n\n"
            "Args:\n"
            "  query (str): Text to search for - keywords or phrases (REQUIRED)\n"
            "  top_k (int): Number of results to return (default 10)\n"
            "  video_ids (list[str] | None): Optional list of video IDs to filter\n"
            "  user_id (str | None): Optional user ID to filter\n"
            "  fuzzy (bool): Enable fuzzy matching for OCR error tolerance (default True)"
        ),
        instructions=(
            "Use when looking for specific words, numbers, or phrases in video frames. "
            "Supports fuzzy matching to handle OCR recognition errors.\n\n"
            "Best paired with: view_ocr_result (after), utility.extract_frames_by_time_window (verification). "
            "Follow up with: view_ocr_result using the handle_id from search results."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
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
            video_ids: Optional list of video IDs to filter (uses bound context if not provided)
            user_id: Optional user ID to filter (uses bound context if not provided)
            fuzzy: Enable fuzzy matching for OCR error tolerance (default True)

        Returns:
            ToolResult with OCR search results
        """
        # Use bound context as defaults
        effective_user_id = user_id or self._user_id
        effective_video_ids = video_ids or self._video_ids

        kwargs = {
            "query": query,
            "top_k": top_k,
            "video_ids": effective_video_ids,
            "user_id": effective_user_id,
            "fuzzy": fuzzy,
        }

        try:
            results = await self.es_client.search_text(
                query=query,
                top_k=top_k,
                video_ids=effective_video_ids,
                user_id=effective_user_id,
                fuzzy=fuzzy,
                highlight=True,
            )
            return self._store_result("search_ocr_text", kwargs, results)
        except Exception as e:
            logger.error(f"[OCRSearchToolkit] search_ocr_text failed: {e}")
            return ToolResult(content=f"Error: OCR text search failed - {str(e)}")

    @tool(
        description=(
            "Get all OCR text extracted from a specific video. "
            "Returns all text found in video frames, sorted by frame order.\n\n"
            "Typical workflow - Comprehensive OCR review:\n"
            "  1. video.list_user_videos - find video ID\n"
            "  2. This tool - get all OCR text for the video\n"
            "  3. view_ocr_result - inspect results with handle_id\n"
            "  4. utility.extract_frames_by_time_window - extract frames at interesting timestamps\n\n"
            "When to use:\n"
            "  - Need all text from a specific video (documentary, lecture, presentation)\n"
            "  - Reviewing all OCR content for a known video\n"
            "  - Finding text patterns within a single video\n\n"
            "Related tools:\n"
            "  - search_ocr_text: Search for specific text across videos\n"
            "  - view_ocr_result: Inspect cached results\n"
            "  - video.get_video_timeline: Get video structure before OCR review\n\n"
            "Args:\n"
            "  video_id (str): Video ID to retrieve OCR for (REQUIRED)\n"
            "  user_id (str | None): Optional user ID to filter by\n"
            "  limit (int): Maximum number of results (default 1000)"
        ),
        instructions=(
            "Use when you need all text from a specific video.\n\n"
            "Best paired with: video.get_video_timeline (before), view_ocr_result (after). "
            "Follow up with: view_ocr_result using the handle_id from results. "
            "Alternative: search_ocr_text for searching specific text across multiple videos."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_ocr_by_video(
        self,
        video_id: str,
        user_id: str | None = None,
        limit: int = 1000,
    ) -> ToolResult:
        """Get all OCR text for a specific video.

        Args:
            video_id: Video ID to retrieve OCR for
            user_id: Optional user ID to filter by (uses bound context if not provided)
            limit: Maximum number of results (default 1000)

        Returns:
            ToolResult with all OCR results for the video
        """
        # Use bound context as default for user_id
        effective_user_id = user_id or self._user_id

        kwargs = {
            "video_id": video_id,
            "user_id": effective_user_id,
            "limit": limit,
        }

        try:
            results = await self.es_client.get_ocr_by_video_id(
                video_id=video_id,
                user_id=effective_user_id,
                limit=limit,
            )
            return self._store_result("get_ocr_by_video", kwargs, results)
        except Exception as e:
            logger.error(f"[OCRSearchToolkit] get_ocr_by_video failed: {e}")
            return ToolResult(content=f"Error: Failed to get OCR for video - {str(e)}")

    @tool(
        description=(
            "View cached OCR search results with different view modes.\n\n"
            "This tool is the companion to all OCR search tools:\n"
            "  - search_ocr_text\n"
            "  - get_ocr_by_video\n\n"
            "Typical workflow:\n"
            "  1. Call search_ocr_text or get_ocr_by_video (results are automatically cached)\n"
            "  2. This tool - inspect cached results with different verbosity levels\n"
            "  3. Use timestamps from results for frame extraction\n\n"
            "View modes:\n"
            "  - 'brief': Top N results with essential info (default)\n"
            "  - 'detailed': Top N results with full details\n"
            "  - 'full': All results with full details\n\n"
            "Related tools:\n"
            "  - search_ocr_text: Keyword search for text in frames\n"
            "  - get_ocr_by_video: Get all OCR for a video\n"
            "  - utility.extract_frames_by_time_window: Extract frames at found timestamps\n"
            "  - view_cache_result (Search toolkit): For video/image/audio results\n"
            "  - view_kg_result (KG toolkit): For knowledge graph results\n\n"
            "Args:\n"
            "  handle_id (str): Handle ID from previous search (REQUIRED)\n"
            "  view_mode (str): 'brief' (top N), 'detailed' (top N detailed), 'full' (all). Default: 'brief'\n"
            "  top_n (int): Number of results to show in brief/detailed mode. Default: 5\n\n"
            "Note: This tool does NOT accept user_id or video_ids parameters."
        ),
        instructions=(
            "Use this to inspect cached OCR results without re-running the search.\n\n"
            "Best paired with: search_ocr_text, get_ocr_by_video (they populate this cache). "
            "Follow up with: utility.extract_frames_by_time_window for visual verification. "
            "Alternative view tools: view_cache_result (Search), view_kg_result (KG)."
        ),
        cache_results=False,
    )
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