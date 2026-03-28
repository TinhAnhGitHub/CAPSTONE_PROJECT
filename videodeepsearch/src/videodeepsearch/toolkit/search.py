from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from agno.tools import tool, Toolkit
from agno.tools.function import ToolResult

from videodeepsearch.clients.storage.qdrant import (
    ImageQdrantClient,
    SegmentQdrantClient,
    AudioQdrantClient,
)
from videodeepsearch.clients.inference import MMBertClient, QwenVLEmbeddingClient, SpladeClient
from videodeepsearch.schemas import ImageInterface, SegmentInterface, AudioInterface
from videodeepsearch.toolkit.common import (
    CacheManager,
    SearchResultContainer,
)


class VideoSearchToolkit(Toolkit):
    """Toolkit for video search with context binding.

    Supports binding user_id and video_ids at initialization time,
    allowing tools to use these as defaults without explicit parameters.
    """

    def __init__(
        self,
        image_qdrant_client: ImageQdrantClient,
        segment_qdrant_client: SegmentQdrantClient,
        audio_qdrant_client: AudioQdrantClient,
        qwenvl_client: QwenVLEmbeddingClient,
        mmbert_client: MMBertClient,
        splade_client: SpladeClient,
        user_id: str | None = None,
        video_ids: list[str] | None = None,
        cache_ttl: int = 1800,
        cache_dir: str | None = None,
    ):
        self.image_client = image_qdrant_client
        self.segment_client = segment_qdrant_client
        self.audio_client = audio_qdrant_client
        self.qwenvl = qwenvl_client
        self.mmbert = mmbert_client
        self.splade = splade_client
        self.cache_ttl = cache_ttl
        self.cache_manager = CacheManager(cache_dir)

        self._user_id = user_id
        self._video_ids = video_ids

        self._result_store: dict[str, SearchResultContainer] = {}

        super().__init__(
            name="Video Search Tools",
            tools=[
                self.get_images_from_caption_query_mmbert,
                self.get_images_from_qwenvl_query,
                self.get_segments_from_event_query_mmbert,
                self.get_segments_from_qwenvl_query,
                self.get_audio_from_query_dense,
                self.get_audio_from_query_hybrid,
                self.view_search_result,
            ],
        )

    def _store_result(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        results: list[ImageInterface] | list[SegmentInterface] | list[AudioInterface],
    ) -> ToolResult:
        if results and isinstance(results[0], ImageInterface):
            result_type = "image"
        elif results and isinstance(results[0], AudioInterface):
            result_type = "audio"
        else:
            result_type = "segment"

        container = SearchResultContainer(
            tool_name=tool_name,
            tool_kwargs=kwargs,
            results=results, #type:ignore
            result_type=result_type,
        )

        handle_id = hashlib.md5(
            json.dumps({"tool": tool_name, "args": kwargs}, sort_keys=True).encode()
        ).hexdigest()[:8]

        self._result_store[handle_id] = container

        content = (
            f"Found {len(results)} {result_type}(s).\n"
            f"Handle ID: {handle_id}\n"
            f"To view: use `view_search_result(handle_id='{handle_id}', view_mode='brief')`\n\n"
            f"{container.get_brief(5)}"
        )

        return ToolResult(content=content)

    @tool(
        description=(
            "Search images by caption using MMBert text embeddings. "
            "Best for queries about WHAT'S HAPPENING or described in captions. "
            "Supports hybrid search combining dense + sparse embeddings.\n\n"
            "Typical workflow (before/after tools):\n"
            "  1. (Optional) llm.enhance_textual_query - expand query into semantic variations\n"
            "  2. This tool - find matching images\n"
            "  3. utility.get_related_asr_from_image - get spoken context around findings\n"
            "Related tools:\n"
            "  - get_images_from_qwenvl_query: Alternative using unified visual embeddings\n"
            "  - get_segments_from_event_query_mmbert: For multi-frame sequences\n"
            "  - get_audio_from_query_dense: For audio/spoken content search\n"
            "  - view_search_result: Inspect cached results by handle_id\n"

            "Args:\n"
            "  caption_query (str): English text describing image content (REQUIRED)\n"
            "  top_k (int): Number of results to return (default 10)\n"
            "  video_ids (list[str] | None): Optional list of video IDs to filter\n"
            "  user_id (str | None): Optional user ID to filter\n"
            "  use_hybrid (bool): Enable hybrid search with sparse encoder (default False)\n"
            "  dense_weight (float): Weight for dense search in hybrid mode (default 0.7)\n"
            "  sparse_weight (float): Weight for sparse search in hybrid mode (default 0.3)"
        ),
        instructions=(
            "Use when query describes EVENTS, ACTIONS, or SCENES. "
            "Query must be in English.\n\n"
            "Best paired with: llm.enhance_textual_query (before), utility.get_related_asr_from_image (after). "
            "Follow up with: view_search_result for detailed inspection. "
            "Alternative: get_images_from_qwenvl_query for unified visual/semantic search."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_images_from_caption_query_mmbert(
        self,
        caption_query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        use_hybrid: bool = False,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> ToolResult:
        effective_user_id = user_id or self._user_id
        effective_video_ids = video_ids or self._video_ids

        kwargs = {
            "caption_query": caption_query,
            "top_k": top_k,
            "video_ids": effective_video_ids,
            "user_id": effective_user_id,
            "use_hybrid": use_hybrid,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
        }

        embeddings = await self.mmbert.ainfer([caption_query])
        if embeddings is None:
            return ToolResult(content="Error: MMBert encoding failed")

        dense_vector = embeddings[0]

        if use_hybrid:
            sparse_vector = (await self.splade.aencode([caption_query]))[0]
            results = await self.image_client.search_image_hybrid_mmbert(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                video_ids=effective_video_ids,
                user_id=effective_user_id,
                limit=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
        else:
            results = await self.image_client.search_image_dense_mmbert(
                query_vector=dense_vector,
                video_ids=effective_video_ids,
                user_id=effective_user_id,
                limit=top_k,
            )

        return self._store_result("get_images_from_caption_query_mmbert", kwargs, results)

    @tool(
        description=(
            "Search images using QwenVL unified multimodal embeddings. "
            "QwenVL provides a unified text-image embedding space, allowing flexible queries. "
            "Works for visual descriptions, semantic captions, or any text query about images.\n\n"
            "Typical workflow (before/after tools):\n"
            "  1. (Optional) llm.enhance_visual_query - generate CLIP-optimized visual variations\n"
            "  2. This tool - search with visual query (each variation is submitted separately)\n"
            "  3. utility.get_related_asr_from_image - get spoken context around findings\n"
            "Related tools:\n"
            "  - get_images_from_caption_query_mmbert: For caption/event-based semantic search\n"
            "  - get_segments_from_qwenvl_query: For segment instead of image search\n"
            "  - get_audio_from_query_dense: For audio/spoken content search\n"
            "  - view_search_result: Inspect cached results by handle_id\n\n"
            "Args:\n"
            "  query (str): Text query describing the image (REQUIRED). Can be visual description, "
            "semantic content, objects/scenes, or any combination\n"
            "  top_k (int): Number of results to return (default 10)\n"
            "  video_ids (list[str] | None): Optional list of video IDs to filter\n"
            "  user_id (str | None): Optional user ID to filter"
        ),
        instructions=(
            "Use for any image search query - visual appearance, semantic content, or mixed. "
            "Query can describe what the image LOOKS LIKE or what's HAPPENING in it.\n\n"
            "Best paired with: llm.enhance_visual_query (before), utility.get_related_asr_from_image (after). "
            "Follow up with: view_search_result for detailed inspection. "
            "Alternative: get_images_from_caption_query_mmbert for caption/event-based search."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_images_from_qwenvl_query(
        self,
        query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
    ) -> ToolResult:
        effective_user_id = user_id or self._user_id
        effective_video_ids = video_ids or self._video_ids

        kwargs = {
            "query": query,
            "top_k": top_k,
            "video_ids": effective_video_ids,
            "user_id": effective_user_id,
        }

        embeddings = await self.qwenvl.ainfer_text([query])
        query_vector = embeddings[0]

        results = await self.image_client.search_image_dense_qwenvl(
            query_vector=query_vector,
            video_ids=effective_video_ids,
            user_id=effective_user_id,
            limit=top_k,
        )

        return self._store_result("get_images_from_qwenvl_query", kwargs, results)

    @tool(
        description=(
            "Search for video segments by event/scene description using MMBert. "
            "Retrieves multi-frame sequences based on semantic similarity to event descriptions. "
            "MMBert provides cleaner language embedding space for text.\n\n"
            "Typical workflow (before/after tools):\n"
            "  1. (Optional) llm.enhance_textual_query - expand event query into semantic variations\n"
            "  2. This tool - find matching video segments (event sequences)\n"
            "  3. utility.get_related_asr_from_segment - get spoken context around found segments\n"
            "Related tools:\n"
            "  - get_segments_from_qwenvl_query: Alternative using unified visual embeddings\n"
            "  - utility.get_adjacent_segments: Navigate to neighboring segments\n"
            "  - get_images_from_caption_query_mmbert: For single-frame image search\n"
            "  - get_audio_from_query_dense: For audio/spoken content search\n"
            "  - view_search_result: Inspect cached results by handle_id\n"

            "Args:\n"
            "  event_query (str): English text describing event/scene (REQUIRED)\n"
            "  top_k (int): Number of results to return (default 10)\n"
            "  video_ids (list[str] | None): Optional list of video IDs to filter\n"
            "  user_id (str | None): Optional user ID to filter\n"
            "  use_hybrid (bool): Enable hybrid search with sparse encoder (default False)\n"
            "  dense_weight (float): Weight for dense search in hybrid mode (default 0.7)\n"
            "  sparse_weight (float): Weight for sparse search in hybrid mode (default 0.3)"
        ),
        instructions=(
            "Use when query describes an EVENT, ACTION, or SCENE (multi-frame sequences).\n\n"
            "Best paired with: llm.enhance_textual_query (before), utility.get_related_asr_from_segment (after). "
            "Follow up with: view_search_result for detailed inspection. "
            "Alternative: get_segments_from_qwenvl_query for unified visual/semantic search."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_segments_from_event_query_mmbert(
        self,
        event_query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        use_hybrid: bool = False,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> ToolResult:
        effective_user_id = user_id or self._user_id
        effective_video_ids = video_ids or self._video_ids

        kwargs = {
            "event_query": event_query,
            "top_k": top_k,
            "video_ids": effective_video_ids,
            "user_id": effective_user_id,
            "use_hybrid": use_hybrid,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
        }

        embeddings = await self.mmbert.ainfer([event_query])
        if embeddings is None:
            return ToolResult(content="Error: MMBert encoding failed")

        query_vector = embeddings[0]

        if use_hybrid:
            sparse_vector = (await self.splade.aencode([event_query]))[0]
            results = await self.segment_client.search_segment_hybrid_mmbert(
                dense_vector=query_vector,
                sparse_vector=sparse_vector,
                video_ids=effective_video_ids,
                user_id=effective_user_id,
                limit=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
        else:
            results = await self.segment_client.search_segment_dense_mmbert(
                query_vector=query_vector,
                video_ids=effective_video_ids,
                user_id=effective_user_id,
                limit=top_k,
            )

        return self._store_result("get_segments_from_event_query_mmbert", kwargs, results)

    @tool(
        description=(
            "Search for video segments using QwenVL unified multimodal embeddings. "
            "Provides unified text-image embedding space for segment retrieval. "
            "Works for visual descriptions, event queries, or any text about segments.\n\n"
            "Typical workflow (before/after tools):\n"
            "  1. (Optional) llm.enhance_visual_query - generate CLIP-optimized visual variations\n"
            "  2. This tool - search with visual query (each variation is submitted separately)\n"
            "  3. utility.get_related_asr_from_segment - get spoken context around found segments\n"
            "Related tools:\n"
            "  - get_segments_from_event_query_mmbert: For caption/event-based semantic search\n"
            "  - utility.get_adjacent_segments: Navigate to neighboring segments\n"
            "  - get_images_from_qwenvl_query: For image instead of segment search\n"
            "  - get_audio_from_query_dense: For audio/spoken content search\n"
            "  - view_search_result: Inspect cached results by handle_id\n\n"
            "Args:\n"
            "  query (str): Text query describing the segment (REQUIRED). Can be visual description, "
            "event/action, scene context, or any combination\n"
            "  top_k (int): Number of results to return (default 10)\n"
            "  video_ids (list[str] | None): Optional list of video IDs to filter\n"
            "  user_id (str | None): Optional user ID to filter"
        ),
        instructions=(
            "Use for any segment search query - visual appearance, event descriptions, or mixed.\n\n"
            "Best paired with: llm.enhance_visual_query (before), utility.get_related_asr_from_segment (after). "
            "Follow up with: view_search_result for detailed inspection. "
            "Alternative: get_segments_from_event_query_mmbert for caption/event-based search."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_segments_from_qwenvl_query(
        self,
        query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
    ) -> ToolResult:
        effective_user_id = user_id or self._user_id
        effective_video_ids = video_ids or self._video_ids

        kwargs = {
            "query": query,
            "top_k": top_k,
            "video_ids": effective_video_ids,
            "user_id": effective_user_id,
        }

        embeddings = await self.qwenvl.ainfer_text([query])
        query_vector = embeddings[0]

        results = await self.segment_client.search_segment_dense_qwenvl(
            query_vector=query_vector,
            video_ids=effective_video_ids,
            user_id=effective_user_id,
            limit=top_k,
        )

        return self._store_result("get_segments_from_qwenvl_query", kwargs, results)

    @tool(
        description=(
            "Search for audio transcripts by text query using MMBert dense embeddings. "
            "Retrieves audio segments based on semantic similarity to spoken content. "
            "Best for finding what was said or discussed in videos.\n\n"
            "Typical workflow (before/after tools):\n"
            "  1. (Optional) llm.enhance_textual_query - expand query into semantic variations\n"
            "  2. This tool - find matching audio transcripts\n"
            "  3. utility.get_related_asr_from_segment - get broader context around found segments\n"
            "Related tools:\n"
            "  - get_audio_from_query_hybrid: For precise keyword + semantic hybrid search\n"
            "  - get_images_from_caption_query_mmbert: For visual search based on spoken content\n"
            "  - kg.search_entities_semantic: For entity-based visual search\n"
            "  - ocr.search_ocr_text: For text visible in frames\n"
            "  - view_search_result: Inspect cached results by handle_id\n\n"
            "Args:\n"
            "  audio_query (str): English text describing spoken content to find (REQUIRED)\n"
            "  top_k (int): Number of results to return (default 10)\n"
            "  video_ids (list[str] | None): Optional list of video IDs to filter\n"
            "  user_id (str | None): Optional user ID to filter"
        ),
        instructions=(
            "Use when query is about SPOKEN CONTENT, SPEECH, or AUDIO.\n\n"
            "Best paired with: llm.enhance_textual_query (before), utility.get_related_asr_from_segment (after). "
            "Follow up with: view_search_result for detailed inspection. "
            "Alternative: get_audio_from_query_hybrid for keyword+semantic hybrid search."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_audio_from_query_dense(
        self,
        audio_query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
    ) -> ToolResult:
        effective_user_id = user_id or self._user_id
        effective_video_ids = video_ids or self._video_ids

        kwargs = {
            "audio_query": audio_query,
            "top_k": top_k,
            "video_ids": effective_video_ids,
            "user_id": effective_user_id,
        }

        embeddings = await self.mmbert.ainfer([audio_query])
        if embeddings is None:
            return ToolResult(content="Error: MMBert encoding failed")

        dense_vector = embeddings[0]

        results = await self.audio_client.search_audio_dense(
            query_vector=dense_vector,
            video_ids=effective_video_ids,
            user_id=effective_user_id,
            limit=top_k,
        )

        return self._store_result("get_audio_from_query_dense", kwargs, results)

    @tool(
        description=(
            "Search for audio transcripts using hybrid dense + sparse search. "
            "Combines MMBert dense embeddings with SPLADE sparse embeddings for better retrieval. "
            "Best for precise keyword matching combined with semantic understanding.\n\n"
            "Typical workflow (before/after tools):\n"
            "  1. (Optional) llm.enhance_textual_query - expand query with semantic variations\n"
            "  2. This tool - find matching audio transcripts with keyword precision\n"
            "  3. utility.get_related_asr_from_segment - get broader context around found segments\n"
            "Related tools:\n"
            "  - get_audio_from_query_dense: For pure semantic search (simpler, faster)\n"
            "  - kg.search_entities_semantic: For entity-based visual search\n"
            "  - ocr.search_ocr_text: For text visible in frames\n"
            "  - view_search_result: Inspect cached results by handle_id\n\n"
            "Args:\n"
            "  audio_query (str): English text describing spoken content to find (REQUIRED)\n"
            "  top_k (int): Number of results to return (default 10)\n"
            "  video_ids (list[str] | None): Optional list of video IDs to filter\n"
            "  user_id (str | None): Optional user ID to filter\n"
            "  dense_weight (float): Weight for dense search (default 0.7)\n"
            "  sparse_weight (float): Weight for sparse search (default 0.3)"
        ),
        instructions=(
            "Use when query contains SPECIFIC KEYWORDS or PHRASES that must be matched exactly.\n\n"
            "Hybrid search combines semantic understanding with keyword precision. "
            "Best paired with: get_audio_from_query_dense (simpler alternative), utility.get_related_asr_from_segment (context). "
            "Follow up with: view_search_result for detailed inspection."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_audio_from_query_hybrid(
        self,
        audio_query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> ToolResult:
        effective_user_id = user_id or self._user_id
        effective_video_ids = video_ids or self._video_ids

        kwargs = {
            "audio_query": audio_query,
            "top_k": top_k,
            "video_ids": effective_video_ids,
            "user_id": effective_user_id,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
        }

        dense_embeddings = await self.mmbert.ainfer([audio_query])
        if dense_embeddings is None:
            return ToolResult(content="Error: MMBert encoding failed")

        dense_vector = dense_embeddings[0]
        sparse_vector = (await self.splade.aencode([audio_query]))[0]

        results = await self.audio_client.search_audio_hybrid(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            video_ids=effective_video_ids,
            user_id=effective_user_id,
            limit=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

        return self._store_result("get_audio_from_query_hybrid", kwargs, results)

    @tool(
        description=(
            "View cached search results by handle_id with different view modes.\n\n"
            "This tool is the companion to all search tools in this toolkit:\n"
            "  - get_images_from_caption_query_mmbert\n"
            "  - get_images_from_qwenvl_query\n"
            "  - get_segments_from_event_query_mmbert\n"
            "  - get_segments_from_qwenvl_query\n"
            "  - get_audio_from_query_dense\n"
            "  - get_audio_from_query_hybrid\n\n"
            "Typical workflow:\n"
            "  1. Call any search tool (results are automatically cached, handle_id is returned)\n"
            "  2. This tool - inspect cached results using the handle_id\n"
            "  3. If satisfied, proceed to context tools (ASR, frame extraction)\n"
            "  4. If not, refine your query and re-run search\n\n"
            "Related tools:\n"
            "  - view_kg_result (KG toolkit): For knowledge graph result inspection\n"
            "  - view_ocr_result (OCR toolkit): For OCR result inspection\n"
            "Args:\n"
            "  handle_id (str): The 8-character handle ID returned by search tools (REQUIRED)\n"
            "  view_mode (str): 'brief', 'detailed', 'statistics', or 'full' (default 'brief')\n"
            "  top_n (int): Number of results to show in brief/detailed mode (default 5)\n"
            "  group_by (str): Grouping strategy for statistics mode: 'video_id' or 'score_bucket' (default 'video_id')"
        ),
        instructions=(
            "Use this to inspect cached results by handle_id without re-running the search.\n\n"
            "The handle_id is returned by all search tools - just copy it and pass it here. "
            "Best paired with: all other search tools (they return handle_ids). "
            "Alternative view tools: view_kg_result (KG), view_ocr_result (OCR)."
        ),
        cache_results=False,
    )
    def view_search_result(
        self,
        handle_id: str,
        view_mode: Literal["brief", "detailed", "statistics", "full"] = "brief",
        top_n: int = 5,
        group_by: Literal["video_id", "score_bucket"] = "video_id",
    ) -> ToolResult:
        container = self._result_store.get(handle_id)
        if not container:
            return ToolResult(
                content=(
                    f"No cached results found for handle_id '{handle_id}'.\n"
                    f"Available handle_ids: {list(self._result_store.keys())}\n"
                    f"Run a search tool first to generate results."
                )
            )

        if view_mode == "brief":
            return ToolResult(content=container.get_brief(top_n))
        elif view_mode == "detailed":
            return ToolResult(content=container.get_detailed(top_n))
        elif view_mode == "statistics":
            return ToolResult(content=container.get_statistics(group_by))
        else:
            return ToolResult(content=container.get_full())


__all__ = ["VideoSearchToolkit"]
