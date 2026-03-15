"""Video Search Toolkit with dependency injection and caching support.

This toolkit provides semantic search capabilities for video content,
supporting both image and segment retrieval with multiple embedding modalities.

Embedding Model Selection Guide:
- MMBert: Prefer for text-only semantic search, document retrieval, sentence similarity.
          Cleaner language embedding space for Vietnamese text.

- QwenVL: Prefer for multimodal scenarios, shared text-image space.
          Use when queries may later include images.

All tools return ToolResult for unified interface.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Literal

from agno.tools import Toolkit, tool
from agno.tools.function import ToolResult
from loguru import logger

from videodeepsearch.clients.storage.qdrant import (
    CaptionQdrantClient,
    ImageQdrantClient,
    SegmentQdrantClient,
)
from videodeepsearch.clients.inference import MMBertClient, QwenVLEmbeddingClient
from videodeepsearch.clients.storage.minio import MinioStorageClient
from videodeepsearch.clients.storage.postgre import PostgresClient
from videodeepsearch.schemas import ImageInterface, SegmentInterface
from videodeepsearch.toolkit.common import (
    CacheManager,
    SearchResultContainer,
    SparseEncoderInterface,
)


def extract_minio_url(url: str) -> tuple[str, str]:
    """Parse MinIO URL to extract bucket and object name.

    Args:
        url: MinIO URL (either http://host/bucket/object or bucket/object format)

    Returns:
        Tuple of (bucket_name, object_name)
    """
    if url.startswith("http"):
        parts = url.split("/", 4)
        if len(parts) >= 5:
            return parts[3], parts[4]
    parts = url.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "videos", url


class VideoSearchToolkit(Toolkit):
    """Toolkit for semantic video search with dependency injection.

    Provides tools for:
    - Visual similarity search (using QwenVL visual embeddings)
    - Caption-based search (using MMBert text or QwenVL multimodal embeddings)
    - Segment search (using various embedding types)

    All tools support caching and return ToolResult for unified interface.
    """

    def __init__(
        self,
        image_qdrant_client: ImageQdrantClient,
        caption_qdrant_client: CaptionQdrantClient,
        segment_qdrant_client: SegmentQdrantClient,
        qwenvl_client: QwenVLEmbeddingClient,
        mmbert_client: MMBertClient,
        minio_client: MinioStorageClient,
        postgres_client: PostgresClient,
        sparse_encoder: SparseEncoderInterface | None = None,
        cache_ttl: int = 1800,
        cache_dir: str | None = None,
    ):
        """Initialize the VideoSearchToolkit.

        Args:
            image_qdrant_client: Client for visual image search
            caption_qdrant_client: Client for caption-based search
            segment_qdrant_client: Client for segment search
            qwenvl_client: QwenVL embedding client for visual/multimodal embeddings
            mmbert_client: MMBert embedding client for text embeddings
            minio_client: MinIO storage client for fetching captions
            postgres_client: PostgreSQL client for metadata queries
            sparse_encoder: Optional sparse encoder for hybrid search
            cache_ttl: Cache time-to-live in seconds (default 30 minutes)
            cache_dir: Optional custom cache directory
        """
        self.image_client = image_qdrant_client
        self.caption_client = caption_qdrant_client
        self.segment_client = segment_qdrant_client
        self.qwenvl = qwenvl_client
        self.mmbert = mmbert_client
        self.storage = minio_client
        self.postgres_client = postgres_client
        self.sparse_encoder = sparse_encoder
        self.cache_ttl = cache_ttl
        self.cache_manager = CacheManager(cache_dir)

        self._result_store: dict[str, SearchResultContainer] = {}

        super().__init__(name="Video Search Tools")

    async def _fetch_caption(self, minio_url: str) -> str:
        """Fetch caption from MinIO JSON file.

        Args:
            minio_url: MinIO URL pointing to the caption JSON file

        Returns:
            Caption string, or empty string if not found
        """
        bucket, object_name = extract_minio_url(minio_url)
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            self.storage.read_json,
            bucket,
            object_name,
        )
        if data is None:
            logger.warning(f"Data is None at {minio_url=}")
            return ""
        caption: str = data.get("caption", "")
        return caption

    async def _fetch_image_caption_from_postgres(self, artifact_id: str) -> str:
        """Fetch image caption from PostgreSQL by finding related ImageCaptionArtifact.

        The artifact_id can be either:
        - ImageEmbeddingArtifact id: We query for its metadata to get image_id,
          then find the ImageCaptionArtifact sibling
        - ImageArtifact id: Directly find the ImageCaptionArtifact child

        Args:
            artifact_id: Artifact ID to start the search from

        Returns:
            Caption string, or empty string if not found
        """
        try:
            artifact = await self.postgres_client.get_artifact(artifact_id)
            if artifact is None:
                logger.warning(f"Artifact {artifact_id} not found in PostgreSQL")
                return ""
            if artifact.artifact_type == "ImageEmbeddingArtifact":
                image_id = artifact.artifact_metadata.get("image_id", "")
                if not image_id:
                    return ""
                return await self.postgres_client.get_caption_by_image_id(image_id)
            elif artifact.artifact_type == "ImageArtifact":
                return await self.postgres_client.get_caption_by_image_id(artifact_id)
            elif artifact.artifact_type == "ImageCaptionArtifact":
                return artifact.artifact_metadata.get("caption", "")
            return ""

        except Exception as e:
            logger.warning(f"Failed to fetch caption from PostgreSQL for {artifact_id}: {e}")
            return ""

    async def _populate_image_captions(
        self,
        results: list[ImageInterface],
        prefer_postgres: bool = False,
    ) -> None:
        """Populate captions for image search results.

        Tries to get caption from Qdrant payload first, then falls back to
        PostgreSQL or MinIO if needed.

        Args:
            results: List of ImageInterface results to populate
            prefer_postgres: If True, always fetch from PostgreSQL (for visual search)
        """
        for item in results:
            # If already has caption from Qdrant and not forcing PostgreSQL, skip
            if item.image_caption and not prefer_postgres:
                continue

            # For visual search results, prefer PostgreSQL
            if prefer_postgres:
                caption = await self._fetch_image_caption_from_postgres(item.id)
                if caption:
                    item.image_caption = caption
                    continue

            # Fallback to MinIO if no caption found
            if not item.image_caption and item.minio_path:
                item.image_caption = await self._fetch_caption(item.minio_path)

    def _store_result(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        results: list[ImageInterface] | list[SegmentInterface],
    ) -> ToolResult:
        """Store results in memory and return ToolResult.

        Args:
            tool_name: Name of the tool
            kwargs: Tool arguments
            results: Search results

        Returns:
            ToolResult with search results and cache instructions
        """
        result_type = "image" if results and isinstance(results[0], ImageInterface) else "segment"

        container = SearchResultContainer(
            tool_name=tool_name,
            tool_kwargs=kwargs,
            results=results,
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

    # =========================================================================
    # Image Search Tools - Visual (QwenVL)
    # =========================================================================

    @tool(
        description=(
            "Search for images using visual similarity based on a natural-language description. "
            "Encodes the input query into a CLIP-style visual embedding (QwenVL) and performs "
            "dense similarity search across indexed video frames."
        ),
        instructions=(
            "Use when query describes what the image LOOKS LIKE, not what's happening. "
            "Query should be in English and visually descriptive. "
            "Best for: objects, scenes, visual patterns, colors, compositions."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_images_from_visual_query(
        self,
        visual_query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
    ) -> ToolResult:
        """Search images by visual description using QwenVL visual embeddings.

        Best for queries describing WHAT THE IMAGE LOOKS LIKE (objects, scenes,
        visual patterns). The query must be in English.

        Args:
            visual_query: English text describing visual appearance
            top_k: Number of results to return (default 10)
            video_ids: Optional list of video IDs to filter
            user_id: Optional user ID to filter

        Returns:
            ToolResult with search results
        """
        kwargs = {
            "visual_query": visual_query,
            "top_k": top_k,
            "video_ids": video_ids,
            "user_id": user_id,
        }

        embeddings = await self.qwenvl.ainfer_text([visual_query])
        query_vector = embeddings[0]

        results = await self.image_client.search_visual(
            query_vector=query_vector,
            video_ids=video_ids,
            user_id=user_id,
            limit=top_k,
        )

        # Fetch captions from PostgreSQL for visual search results
        await self._populate_image_captions(results, prefer_postgres=True)

        return self._store_result("get_images_from_visual_query", kwargs, results)

    # =========================================================================
    # Image Search Tools - Caption (MMBert - Text)
    # =========================================================================

    @tool(
        description=(
            "Search images by Vietnamese caption text using MMBert semantic embeddings. "
            "MMBert provides cleaner language embedding space optimized for text. "
            "Performs dense semantic search over image captions."
        ),
        instructions=(
            "Use when query is in english describing image content/meaning. "
            "Best for: semantic search, document retrieval, sentence similarity. "
            "Prefer MMBert when your system is text-only."
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
        """Search images by Vietnamese caption using MMBert text embeddings.

        Best for queries about WHAT'S HAPPENING or described in captions.
        The caption query must be in Vietnamese.

        Prefer MMBert for:
        - Semantic search in Vietnamese
        - Document retrieval
        - Sentence similarity
        - Text-only systems

        Args:
            caption_query: Vietnamese text describing image content
            top_k: Number of results to return (default 10)
            video_ids: Optional list of video IDs to filter
            user_id: Optional user ID to filter
            use_hybrid: Enable hybrid search with sparse encoder (requires sparse_encoder)
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

        if use_hybrid and self.sparse_encoder:
            sparse_vector = (await self.sparse_encoder.encode([caption_query]))[0]
            results = await self.caption_client.search_hybrid_text(
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                video_ids=video_ids,
                user_id=user_id,
                limit=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
        else:
            results = await self.caption_client.search_text(
                query_vector=dense_vector,
                video_ids=video_ids,
                user_id=user_id,
                limit=top_k,
            )

        await self._populate_image_captions(results)

        return self._store_result("get_images_from_caption_query_mmbert", kwargs, results)

    # =========================================================================
    # Image Search Tools - Caption (QwenVL - Multimodal)
    # =========================================================================

    @tool(
        description=(
            "Search images by caption text using QwenVL multimodal embeddings. "
            "QwenVL provides a unified text-image embedding space. "
            "Use when your system also uses image embeddings or queries may later include images."
        ),
        instructions=(
            "Use when you need shared text-image embedding space. "
            "Best for: multimodal systems, when queries might later include images. "
            "Query can be in Vietnamese or English."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_images_from_caption_query_qwenvl(
        self,
        caption_query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
    ) -> ToolResult:
        """Search images by caption using QwenVL multimodal embeddings.

        Best for queries requiring a unified text-image embedding space.

        Prefer QwenVL for:
        - Multimodal systems with both text and images
        - Shared text-image embedding space
        - Queries that may later include images

        Args:
            caption_query: Text describing image content (Vietnamese or English)
            top_k: Number of results to return (default 10)
            video_ids: Optional list of video IDs to filter
            user_id: Optional user ID to filter

        Returns:
            ToolResult with search results
        """
        kwargs = {
            "caption_query": caption_query,
            "top_k": top_k,
            "video_ids": video_ids,
            "user_id": user_id,
        }

        embeddings = await self.qwenvl.ainfer_text([caption_query])
        query_vector = embeddings[0]

        results = await self.caption_client.search_multimodal(
            query_vector=query_vector,
            video_ids=video_ids,
            user_id=user_id,
            limit=top_k,
        )

        await self._populate_image_captions(results)

        return self._store_result("get_images_from_caption_query_qwenvl", kwargs, results)

    # =========================================================================
    # Segment Search Tools - Event (MMBert - Vietnamese Text)
    # =========================================================================

    @tool(
        description=(
            "Search for video segments by Vietnamese event/scene description using MMBert. "
            "Retrieves multi-frame sequences based on semantic similarity to event descriptions. "
            "MMBert provides cleaner language embedding space for Vietnamese text."
        ),
        instructions=(
            "Use when query describes an EVENT, ACTION, or SCENE in Vietnamese. "
            "Best for: finding continuous video sequences, temporal events, conversations. "
            "Prefer MMBert for text-only semantic search in Vietnamese."
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
    ) -> ToolResult:
        """Search segments by Vietnamese event description using MMBert.

        Best for queries about EVENTS, ACTIONS, or SCENES (multi-frame sequences).
        The event query must be in Vietnamese.

        Prefer MMBert for:
        - Semantic search in Vietnamese
        - Event/scene retrieval based on text descriptions

        Args:
            event_query: Vietnamese text describing event/scene
            top_k: Number of results to return (default 10)
            video_ids: Optional list of video IDs to filter
            user_id: Optional user ID to filter

        Returns:
            ToolResult with search results
        """
        kwargs = {
            "event_query": event_query,
            "top_k": top_k,
            "video_ids": video_ids,
            "user_id": user_id,
        }

        embeddings = await self.mmbert.ainfer([event_query])
        if embeddings is None:
            return ToolResult(content="Error: MMBert encoding failed")

        query_vector = embeddings[0]

        results = await self.segment_client.search_segment(
            query_vector=query_vector,
            video_ids=video_ids,
            user_id=user_id,
            limit=top_k,
        )

        return self._store_result("get_segments_from_event_query_mmbert", kwargs, results)

    # =========================================================================
    # Segment Search Tools - Event (QwenVL - Multimodal)
    # =========================================================================

    @tool(
        description=(
            "Search for video segments using QwenVL multimodal embeddings. "
            "Provides unified text-image embedding space for segment retrieval. "
            "Use when your system also uses image embeddings."
        ),
        instructions=(
            "Use when you need shared text-image embedding space for segments. "
            "Best for: multimodal systems, when queries might later include images. "
            "Query can be in Vietnamese or English."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_segments_from_event_query_qwenvl(
        self,
        event_query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
    ) -> ToolResult:
        """Search segments by event description using QwenVL multimodal embeddings.

        Best for queries requiring a unified text-image embedding space.

        Prefer QwenVL for:
        - Multimodal systems with both text and images
        - Shared text-image embedding space

        Args:
            event_query: Text describing event/scene (Vietnamese or English)
            top_k: Number of results to return (default 10)
            video_ids: Optional list of video IDs to filter
            user_id: Optional user ID to filter

        Returns:
            ToolResult with search results
        """
        kwargs = {
            "event_query": event_query,
            "top_k": top_k,
            "video_ids": video_ids,
            "user_id": user_id,
        }

        embeddings = await self.qwenvl.ainfer_text([event_query])
        query_vector = embeddings[0]

        results = await self.segment_client.search_segment(
            query_vector=query_vector,
            video_ids=video_ids,
            user_id=user_id,
            limit=top_k,
        )

        return self._store_result("get_segments_from_event_query_qwenvl", kwargs, results)

    # =========================================================================
    # Segment Search Tools - Visual (QwenVL)
    # =========================================================================

    @tool(
        description=(
            "Search for video segments using visual similarity. "
            "Encodes the visual description into QwenVL embedding and searches "
            "for segments with similar visual content."
        ),
        instructions=(
            "Use when query describes what the SCENE LOOKS LIKE visually. "
            "Query should be in English and visually descriptive. "
            "Best for: visual patterns, scene appearances, object-based segment retrieval."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_segments_from_visual_query(
        self,
        visual_query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
    ) -> ToolResult:
        """Search segments by visual description using QwenVL embeddings.

        Best for queries describing WHAT THE SCENE LOOKS LIKE visually.
        The visual query must be in English.

        Args:
            visual_query: English text describing visual appearance
            top_k: Number of results to return (default 10)
            video_ids: Optional list of video IDs to filter
            user_id: Optional user ID to filter

        Returns:
            ToolResult with search results
        """
        kwargs = {
            "visual_query": visual_query,
            "top_k": top_k,
            "video_ids": video_ids,
            "user_id": user_id,
        }

        embeddings = await self.qwenvl.ainfer_text([visual_query])
        query_vector = embeddings[0]

        results = await self.segment_client.search_segment(
            query_vector=query_vector,
            video_ids=video_ids,
            user_id=user_id,
            limit=top_k,
        )

        return self._store_result("get_segments_from_visual_query", kwargs, results)

    # =========================================================================
    # Multi-Modal Search (Combined Visual + Caption)
    # =========================================================================

    @tool(
        description=(
            "Search images using combined visual + caption signals (multimodal fusion). "
            "Performs multimodal retrieval by fusing visual and semantic search signals. "
            "Enables finding images that match both visual appearance AND semantic description."
        ),
        instructions=(
            "Use when you need to match BOTH visual appearance AND semantic meaning. "
            "Visual query in English, caption query in Vietnamese. "
            "Best for: complex queries requiring multi-faceted matching."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_images_from_multimodal_query(
        self,
        visual_query: str,
        caption_query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        visual_weight: float = 0.5,
        caption_weight: float = 0.5,
    ) -> ToolResult:
        """Search images using combined visual and caption signals.

        Best for queries requiring BOTH visual appearance AND semantic meaning.
        Visual query in English, caption query in Vietnamese.

        Args:
            visual_query: English text describing visual appearance
            caption_query: Vietnamese text describing content
            top_k: Number of results to return (default 10)
            video_ids: Optional list of video IDs to filter
            user_id: Optional user ID to filter
            visual_weight: Weight for visual search (default 0.5)
            caption_weight: Weight for caption search (default 0.5)

        Returns:
            ToolResult with search results
        """
        kwargs = {
            "visual_query": visual_query,
            "caption_query": caption_query,
            "top_k": top_k,
            "video_ids": video_ids,
            "user_id": user_id,
            "visual_weight": visual_weight,
            "caption_weight": caption_weight,
        }

        visual_emb = await self.qwenvl.ainfer_text([visual_query])
        caption_emb = await self.qwenvl.ainfer_text([caption_query])

        results = await self.image_client.search_multi_dense(
            vectors=[
                ("image_dense", visual_emb[0]),
                ("caption_mm_dense", caption_emb[0]),
            ],
            weights=[visual_weight, caption_weight],
            limit=top_k,
            query_filter=self.image_client.build_filter(video_ids, user_id),
        )

        # Fetch captions from PostgreSQL for multimodal search results
        await self._populate_image_captions(results, prefer_postgres=True)

        return self._store_result("get_images_from_multimodal_query", kwargs, results)

    # =========================================================================
    # Cache Retrieval Tool
    # =========================================================================

    @tool(
        description="View cached search results with different view modes.",
        cache_results=False,
    )
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


__all__ = ["VideoSearchToolkit"]