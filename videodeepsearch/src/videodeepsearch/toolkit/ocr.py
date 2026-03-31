from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from agno.tools import Toolkit, tool
from loguru import logger

from videodeepsearch.clients.storage.elasticsearch import (
    ElasticsearchOCRClient,
    OCRSearchResult,
)
from videodeepsearch.clients.inference import MMBertClient
from videodeepsearch.toolkit.common import CacheManager


class OCRSearchToolkit(Toolkit):
    def __init__(
        self,
        es_ocr_client: ElasticsearchOCRClient,
        mmbert_client: MMBertClient | None = None,
        user_id: str | None = None,
        video_ids: list[str] | None = None,
        cache_ttl: int = 1800,
        cache_dir: str | None = None,
    ):
        self.es_client = es_ocr_client
        self.mmbert = mmbert_client
        self.cache_ttl = cache_ttl
        self.cache_manager = CacheManager(cache_dir)
        self._result_store: dict[str, list[OCRSearchResult]] = {}

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
    ) -> dict[str, Any]:
        handle_id = hashlib.md5(
            json.dumps({"tool": tool_name, "args": kwargs}, sort_keys=True).encode()
        ).hexdigest()[:8]

        self._result_store[handle_id] = results

        return self._get_brief(results, 5, handle_id)

    def _get_brief(self, results: list[OCRSearchResult], top_n: int = 5, handle_id: str | None = None) -> dict[str, Any]:
        """Get brief representation of top N results as JSON dict."""
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)[:top_n]

        return {
            "view_mode": "brief",
            "total": len(results),
            "top_n": min(top_n, len(sorted_results)),
            "handle_id": handle_id,
            "results": [r.model_dump() for r in sorted_results],
        }

    def _get_detailed(self, results: list[OCRSearchResult], top_n: int = 5) -> dict[str, Any]:
        """Get detailed representation of top N results as JSON dict."""
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)[:top_n]

        return {
            "view_mode": "detailed",
            "total": len(results),
            "top_n": min(top_n, len(sorted_results)),
            "results": [r.model_dump() for r in sorted_results],
        }

    def _get_full(self, results: list[OCRSearchResult]) -> dict[str, Any]:
        """Get full results as JSON dict."""
        return {
            "view_mode": "full",
            "total": len(results),
            "results": [r.model_dump() for r in results],
        }

    @tool(
        description=(
            "Search for text in video frames using keyword matching (BM25). "
            "Best for finding specific words, phrases, or exact text matches. "
            "Supports fuzzy matching for handling OCR errors.\n\n"
            "Typical workflow - OCR search and verification:\n"
            "  1. This tool - find frames containing specific text\n"
            "  2. view_ocr_result - inspect detailed results with handle_id\n"
            "  4. utility.get_related_asr_from_image - get spoken context around found text\n\n"
            "When to use:\n"
            "  - Looking for specific words, numbers, or phrases in video frames\n"
            "  - Searching for labels, captions, signs, or on-screen text\n"
            "  - Finding documents, slides, or text-heavy content\n\n"
            "Related tools:\n"
            "  - get_ocr_by_video: Get all OCR text for a specific video\n"
            "  - view_ocr_result: Inspect cached search results\n"
            "  - search.get_images_from_caption_query_mmbert: For image content search (not text)\n"
            "  - utility.get_related_asr_from_image: Get spoken context around found text\n\n"
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
    ) -> dict[str, Any]:
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
            return {"error": f"OCR text search failed - {str(e)}"}

    @tool(
        description=(
            "Get all OCR text extracted from a specific video. "
            "Returns all text found in video frames, sorted by frame order.\n\n"
            "Typical workflow - Comprehensive OCR review:\n"
            "  1. video.list_user_videos - find video ID\n"
            "  2. This tool - get all OCR text for the video\n"
            "  3. view_ocr_result - inspect results with handle_id\n"
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
    ) -> dict[str, Any]:
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
            return {"error": f"Failed to get OCR for video - {str(e)}"}

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
            "Alternative view tools: view_cache_result (Search), view_kg_result (KG)."
        ),
        cache_results=False,
    )
    def view_ocr_result(
        self,
        handle_id: str,
        view_mode: Literal["brief", "detailed", "full"] = "brief",
        top_n: int = 5,
    ) -> dict[str, Any]:
        if handle_id not in self._result_store:
            return {
                "error": f"No cached results found for handle_id '{handle_id}'",
                "available_handle_ids": list(self._result_store.keys()),
            }

        results = self._result_store[handle_id]

        if view_mode == "brief":
            return self._get_brief(results, top_n)
        elif view_mode == "detailed":
            return self._get_detailed(results, top_n)
        else:
            return self._get_full(results)


__all__ = ["OCRSearchToolkit"]
