from __future__ import annotations


from agno.tools import Toolkit, tool
from agno.tools.function import ToolResult

from loguru import logger

from videodeepsearch.clients.storage.elasticsearch import (
    ElasticsearchOCRClient,
    OCRSearchResult,
)
from videodeepsearch.clients.inference import MMBertClient
from videodeepsearch.tracing import traced_tool


class OCRSearchToolkit(Toolkit):
    def __init__(
        self,
        es_ocr_client: ElasticsearchOCRClient,
        mmbert_client: MMBertClient,
        user_id: str,
        video_ids: list[str],
    ):
        self.es_client = es_ocr_client
        self.mmbert = mmbert_client

        self._user_id = user_id
        self._video_ids = video_ids

        super().__init__(
            name="OCR Search Tools",
            tools=[
                self.search_ocr_text,
                self.get_ocr_by_video,
            ],
        )

    def get_full_display (self, results: list[OCRSearchResult]) -> str:
        """Get full results as JSON dict."""
        
        content_return = [
            "Top Result: "
        ]
        
        for item in results:
            content_return.append(item.detailed_representation())
        
        return "\n\n".join(content_return)

    @tool(
        description=(
            "Search for text in video frames using keyword matching (BM25). "
            "Best for finding specific words, phrases, or exact text matches. "
            "Supports fuzzy matching for handling OCR errors.\n\n"
            "Typical workflow - OCR search and verification:\n"
            "  1. This tool - find frames containing specific text\n"
            "  2. utility.get_related_asr_from_image - get spoken context around found text\n\n"
            "When to use:\n"
            "  - Looking for specific words, numbers, or phrases in video frames\n"
            "  - Searching for labels, captions, signs, or on-screen text\n"
            "  - Finding documents, slides, or text-heavy content\n\n"
            "Related tools:\n"
            "  - get_ocr_by_video: Get all OCR text for a specific video\n"
            "  - search.get_images_from_caption_query_mmbert: For image content search (not text)\n"
            "  - utility.get_related_asr_from_image: Get spoken context around found text\n\n"
            "Args:\n"
            "  query (str): Text to search for - keywords or phrases (REQUIRED)\n"
            "  top_k (int): Number of results to return (default 10)\n"
            "  video_ids (list[str] | None): Optional list of video IDs to filter\n"
            "  fuzzy (bool): Enable fuzzy matching for OCR error tolerance (default True)"
        ),
        instructions=(
            "Use when looking for specific words, numbers, or phrases in video frames. "
            "Supports fuzzy matching to handle OCR recognition errors.\n\n"
        ),
    )
    @traced_tool()
    async def search_ocr_text(
        self,
        query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        fuzzy: bool = True,
    ) -> ToolResult:
        effective_video_ids = video_ids or self._video_ids



        try:
            results = await self.es_client.search_text(
                query=query,
                top_k=top_k,
                video_ids=effective_video_ids,
                user_id=self._user_id,
                fuzzy=fuzzy,
                highlight=True,
            )
            return ToolResult(content=self.get_full_display(results))
        except Exception as e:
            logger.error(f"[OCRSearchToolkit] search_ocr_text failed: {e}")
            return ToolResult(content=f"OCR text search failed - {str(e)}")

    @tool(
        description=(
            "Get all OCR text extracted from a specific video. "
            "Returns all text found in video frames, sorted by frame order.\n\n"
            "Typical workflow - Comprehensive OCR review:\n"
            "  1. video.list_user_videos - find video ID\n"
            "  2. This tool - get all OCR text for the video\n"
            "When to use:\n"
            "  - Need all text from a specific video (documentary, lecture, presentation)\n"
            "  - Reviewing all OCR content for a known video\n"
            "  - Finding text patterns within a single video\n\n"
            "Related tools:\n"
            "  - search_ocr_text: Search for specific text across videos\n"
            "  - video.get_video_timeline: Get video structure before OCR review\n\n"
            "Args:\n"
            "  video_id (str): Video ID to retrieve OCR for (REQUIRED)\n"
            "  limit (int): Maximum number of results (default 1000)"
        ),
        instructions=(
            "Use when you need all text from a specific video.\n\n"
            "Best paired with: video.get_video_timeline (before)"
            "Alternative: search_ocr_text for searching specific text across multiple videos."
        ),
    )
    @traced_tool()
    async def get_ocr_by_video(
        self,
        video_id: str,
        limit: int = 1000,
    ) -> ToolResult:

        try:
            results = await self.es_client.get_ocr_by_video_id(
                video_id=video_id,
                user_id=self._user_id,
                limit=limit,
            )
            return ToolResult(content=self.get_full_display(results))
        except Exception as e:
            logger.error(f"[OCRSearchToolkit] get_ocr_by_video failed: {e}")
            return ToolResult(content=f"Failed to get OCR for video - {str(e)}")


__all__ = ["OCRSearchToolkit"]
