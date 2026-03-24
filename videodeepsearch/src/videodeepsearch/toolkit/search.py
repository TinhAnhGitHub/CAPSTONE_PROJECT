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

    def __init__(
        self,
        image_qdrant_client: ImageQdrantClient,
        segment_qdrant_client: SegmentQdrantClient,
        audio_qdrant_client: AudioQdrantClient,
        qwenvl_client: QwenVLEmbeddingClient,
        mmbert_client: MMBertClient,
        splade_client: SpladeClient,
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

        self._result_store: dict[str, SearchResultContainer] = {}

        super().__init__(name="Video Search Tools")

    def _store_result(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        results: list[ImageInterface] | list[SegmentInterface] | list[AudioInterface],
    ) -> ToolResult:
        """Store results in memory and return ToolResult.

        Args:
            tool_name: Name of the tool
            kwargs: Tool arguments
            results: Search results

        Returns:
            ToolResult with search results and cache instructions
        """
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
            f"To view: use `view_cache_result(tool_name='{tool_name}', args={kwargs}, view_mode='brief')`\n\n"
            f"{container.get_brief(5)}"
        )

        return ToolResult(content=content)

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
        """Search images by caption using MMBert text embeddings.

        Best for queries about WHAT'S HAPPENING or described in captions.
        The caption query must be in English.

        Prefer MMBert for:
        - Semantic search in English
        - Document retrieval
        - Sentence similarity
        - Text-only systems

        Args:
            caption_query: English text describing image content
            top_k: Number of results to return (default 10)
            video_ids: Optional list of video IDs to filter
            user_id: Optional user ID to filter
            use_hybrid: Enable hybrid search with sparse encoder (requires splade_client)
            dense_weight: Weight for dense search in hybrid mode (default 0.7)
            sparse_weight: Weight for sparse search in hybrid mode (default 0.3)

        Returns:
            ToolResult with search results
        """
        kwargs = {
            "caption_query": caption_query,
            "top_k": top_k,
            "video_ids": video_ids,
            "user_id": user_id,
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
                video_ids=video_ids,
                user_id=user_id,
                limit=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
        else:
            results = await self.image_client.search_image_dense_mmbert(
                query_vector=dense_vector,
                video_ids=video_ids,
                user_id=user_id,
                limit=top_k,
            )

        return self._store_result("get_images_from_caption_query_mmbert", kwargs, results)

    # @tool(
    #     description=(
    #         "Search images using QwenVL unified multimodal embeddings. "
    #         "QwenVL provides a unified text-image embedding space, allowing flexible queries. "
    #         "Works for visual descriptions, semantic captions, or any text query about images."
    #     ),
    #     instructions=(
    #         "Use for any image search query - visual appearance, semantic content, or mixed. "
    #         "Query can describe what the image LOOKS LIKE or what's HAPPENING in it. "
    #         "Best for: visual search, caption search, multimodal retrieval."
    #     ),
    #     cache_results=True,
    #     cache_ttl=1800,
    # )
    async def get_images_from_qwenvl_query(
        self,
        query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
    ) -> ToolResult:
        """Search images using QwenVL unified multimodal embeddings.

        QwenVL provides a unified embedding space for both visual and semantic queries.
        The query can be any text describing the image - visual appearance, content,
        events, objects, or semantic meaning.

        Args:
            query: Text query describing the image. Can be:
                - Visual description (e.g., "a red car on a sunny street")
                - Semantic content (e.g., "people having a conversation")
                - Objects/scenes (e.g., "kitchen with modern appliances")
                - Any combination of the above
            top_k: Number of results to return (default 10)
            video_ids: Optional list of video IDs to filter
            user_id: Optional user ID to filter

        Returns:
            ToolResult with search results
        """
        kwargs = {
            "query": query,
            "top_k": top_k,
            "video_ids": video_ids,
            "user_id": user_id,
        }

        embeddings = await self.qwenvl.ainfer_text([query])
        query_vector = embeddings[0]

        results = await self.image_client.search_image_dense_qwenvl(
            query_vector=query_vector,
            video_ids=video_ids,
            user_id=user_id,
            limit=top_k,
        )

        return self._store_result("get_images_from_qwenvl_query", kwargs, results)

    # @tool(
    #     description=(
    #         "Search for video segments by event/scene description using MMBert. "
    #         "Retrieves multi-frame sequences based on semantic similarity to event descriptions. "
    #         "MMBert provides cleaner language embedding space for text."
    #     ),
    #     instructions=(
    #         "Use when query describes an EVENT, ACTION, or SCENE. "
    #         "Best for: finding continuous video sequences, temporal events, conversations. "
    #         "Prefer MMBert for text-only semantic search."
    #     ),
    #     cache_results=True,
    #     cache_ttl=1800,
    # )
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
        """Search segments by event description using MMBert.

        Best for queries about EVENTS, ACTIONS, or SCENES (multi-frame sequences).
        The event query must be in English.

        Prefer MMBert for:
        - Semantic search in English
        - Event/scene retrieval based on text descriptions

        Args:
            event_query: English text describing event/scene
            top_k: Number of results to return (default 10)
            video_ids: Optional list of video IDs to filter
            user_id: Optional user ID to filter
            use_hybrid: Enable hybrid search with sparse encoder (requires splade_client)
            dense_weight: Weight for dense search in hybrid mode (default 0.7)
            sparse_weight: Weight for sparse search in hybrid mode (default 0.3)

        Returns:
            ToolResult with search results
        """
        kwargs = {
            "event_query": event_query,
            "top_k": top_k,
            "video_ids": video_ids,
            "user_id": user_id,
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
                video_ids=video_ids,
                user_id=user_id,
                limit=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
        else:
            results = await self.segment_client.search_segment_dense_mmbert(
                query_vector=query_vector,
                video_ids=video_ids,
                user_id=user_id,
                limit=top_k,
            )

        return self._store_result("get_segments_from_event_query_mmbert", kwargs, results)

    # @tool(
    #     description=(
    #         "Search for video segments using QwenVL unified multimodal embeddings. "
    #         "Provides unified text-image embedding space for segment retrieval. "
    #         "Works for visual descriptions, event queries, or any text about segments."
    #     ),
    #     instructions=(
    #         "Use for any segment search query - visual appearance, event descriptions, or mixed. "
    #         "Query can describe what the SCENE LOOKS LIKE or what's HAPPENING in it. "
    #         "Best for: visual search, event retrieval, multimodal segment queries."
    #     ),
    #     cache_results=True,
    #     cache_ttl=1800,
    # )
    async def get_segments_from_qwenvl_query(
        self,
        query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
    ) -> ToolResult:
        """Search segments using QwenVL unified multimodal embeddings.

        QwenVL provides a unified embedding space for both visual and semantic queries.
        The query can be any text describing the segment - visual appearance, events,
        actions, or semantic meaning.

        Args:
            query: Text query describing the segment. Can be:
                - Visual description (e.g., "outdoor scene with trees and sunlight")
                - Event/action (e.g., "people discussing project plans")
                - Scene context (e.g., "meeting room with whiteboard")
                - Any combination of the above
            top_k: Number of results to return (default 10)
            video_ids: Optional list of video IDs to filter
            user_id: Optional user ID to filter

        Returns:
            ToolResult with search results
        """
        kwargs = {
            "query": query,
            "top_k": top_k,
            "video_ids": video_ids,
            "user_id": user_id,
        }

        embeddings = await self.qwenvl.ainfer_text([query])
        query_vector = embeddings[0]

        results = await self.segment_client.search_segment_dense_qwenvl(
            query_vector=query_vector,
            video_ids=video_ids,
            user_id=user_id,
            limit=top_k,
        )

        return self._store_result("get_segments_from_qwenvl_query", kwargs, results)

    @tool(
        description=(
            "Search for audio transcripts by text query using MMBert dense embeddings. "
            "Retrieves audio segments based on semantic similarity to spoken content. "
            "Best for finding what was said or discussed in videos."
        ),
        instructions=(
            "Use when query is about SPOKEN CONTENT, SPEECH, or AUDIO. "
            "Best for: finding specific phrases, topics discussed, conversations. "
            "Prefer MMBert for text-only semantic search over audio transcripts."
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
        """Search audio transcripts by text query using dense embeddings.

        Best for queries about SPOKEN CONTENT in videos.
        The audio query must be in English.

        Args:
            audio_query: English text describing spoken content to find
            top_k: Number of results to return (default 10)
            video_ids: Optional list of video IDs to filter
            user_id: Optional user ID to filter

        Returns:
            ToolResult with search results
        """
        kwargs = {
            "audio_query": audio_query,
            "top_k": top_k,
            "video_ids": video_ids,
            "user_id": user_id,
        }

        embeddings = await self.mmbert.ainfer([audio_query])
        if embeddings is None:
            return ToolResult(content="Error: MMBert encoding failed")

        dense_vector = embeddings[0]

        results = await self.audio_client.search_audio_dense(
            query_vector=dense_vector,
            video_ids=video_ids,
            user_id=user_id,
            limit=top_k,
        )

        return self._store_result("get_audio_from_query_dense", kwargs, results)

    # @tool(
    #     description=(
    #         "Search for audio transcripts using hybrid dense + sparse search. "
    #         "Combines MMBert dense embeddings with SPLADE sparse embeddings for better retrieval. "
    #         "Best for precise keyword matching combined with semantic understanding."
    #     ),
    #     instructions=(
    #         "Use when query contains SPECIFIC KEYWORDS or PHRASES that must be matched exactly. "
    #         "Hybrid search combines semantic understanding with keyword precision. "
    #         "Best for: finding exact phrases, technical terms, named entities in audio."
    #     ),
    #     cache_results=True,
    #     cache_ttl=1800,
    # )
    async def get_audio_from_query_hybrid(
        self,
        audio_query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> ToolResult:
        """Search audio transcripts using hybrid dense + sparse search.

        Best for queries that need both semantic understanding and keyword matching.
        The audio query must be in English.

        Args:
            audio_query: English text describing spoken content to find
            top_k: Number of results to return (default 10)
            video_ids: Optional list of video IDs to filter
            user_id: Optional user ID to filter
            dense_weight: Weight for dense search (default 0.7)
            sparse_weight: Weight for sparse search (default 0.3)

        Returns:
            ToolResult with search results
        """
        kwargs = {
            "audio_query": audio_query,
            "top_k": top_k,
            "video_ids": video_ids,
            "user_id": user_id,
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
            video_ids=video_ids,
            user_id=user_id,
            limit=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

        return self._store_result("get_audio_from_query_hybrid", kwargs, results)

    # =========================================================================
    # Cache Retrieval Tool
    # =========================================================================

    # @tool(
    #     description="View cached search results with different view modes.",
    #     cache_results=False,
    # )
    def view_cache_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        view_mode: Literal["brief", "detailed", "statistics", "full"] = "brief",
        top_n: int = 5,
        group_by: Literal["video_id", "score_bucket"] = "video_id",
    ) -> ToolResult:
        """View previously cached search results.

        Use this to inspect cached results without re-running the search.
        Supports different view modes to control output verbosity.

        Args:
            tool_name: Name of the tool that generated the results
            args: Arguments that were passed to the tool
            view_mode: 'brief' (top 5), 'detailed' (top 5 detailed),
                      'statistics' (grouped stats), 'full' (all results)
            top_n: Number of results to show in brief/detailed mode
            group_by: Grouping strategy for statistics mode

        Returns:
            ToolResult with formatted view of cached results
        """
        handle_id = hashlib.md5(
            json.dumps({"tool": tool_name, "args": args}, sort_keys=True).encode()
        ).hexdigest()[:8]
        print(f"{handle_id=}")

        if handle_id in self._result_store:
            container = self._result_store[handle_id]
        else:
            _, hit = self.cache_manager.get_cached_result(tool_name, args)
            if not hit:
                return ToolResult(
                    content=(
                        f"No cached results found for {tool_name} with args {args}.\n"
                        f"Run the search tool first to generate results."
                    )
                )

            return ToolResult(
                content=(
                    f"Found cached result for {tool_name}.\n"
                    f"Note: Results are stored as strings. "
                    f"Re-run the search to get structured results for viewing."
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


    def view_cache_result_from_handle_id(
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
                    f"Found cached result for {handle_id=}.\n"
                    f"Note: Results are stored as strings. "
                    f"Re-run the search to get structured results for viewing."
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

    def list_cached_results(self) -> ToolResult:
        """List all available handle IDs with a brief summary.

        Returns:
            ToolResult with list of cached results
        """
        if not self._result_store:
            return ToolResult(content="No cached results available.")

        lines = ["=== Cached Search Results ===", ""]

        for handle_id, container in self._result_store.items():
            lines.append(f"Handle ID: {handle_id}")
            lines.append(f"  Tool: {container.tool_name}")
            lines.append(f"  Type: {container.result_type}")
            lines.append(f"  Total: {len(container.results)} result(s)")
            lines.append(f"  Args: {container.tool_kwargs}")
            lines.append("")

        return ToolResult(content="\n".join(lines))




__all__ = ["VideoSearchToolkit"]