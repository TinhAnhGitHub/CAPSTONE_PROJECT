from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from agno.tools import Toolkit, tool
from agno.tools.function import ToolResult
from arango.database import StandardDatabase
from loguru import logger

from videodeepsearch.clients.inference import MMBertClient
from videodeepsearch.toolkit.common import CacheManager


class KGSearchToolkit(Toolkit):
    def __init__(
        self,
        arango_db: StandardDatabase,
        mmbert_client: MMBertClient | None = None,
        user_id: str | None = None,
        video_ids: list[str] | None = None,
        graph_name: str = "video_knowledge_graph",
        search_view: str = "video_kg_search_view",
        cache_ttl: int = 1800,
        cache_dir: str | None = None,
    ):
        self.db = arango_db
        self.mmbert = mmbert_client
        self.graph_name = graph_name
        self.search_view = search_view
        self.cache_ttl = cache_ttl
        self.cache_manager = CacheManager(cache_dir)
        self._result_store: dict[str, list[dict[str, Any]]] = {}

        self._user_id = user_id
        self._video_ids = video_ids

        super().__init__(
            name="Knowledge Graph Search Tools",
            tools=[
                self.search_entities_semantic,
                self.search_events,
                self.search_micro_events,
                self.search_communities,
                self.traverse_from_entity,
                self.multi_granularity_search,
                self.search_bm25,
                self.triple_hybrid_search,
                self.retrieve_for_rag,
                self.view_kg_result,
            ],
        )

    async def _encode_query_async(self, query: str) -> list[float] | None:
        if not self.mmbert:
            return None
        try:
            result = await self.mmbert.ainfer([query])
            if result:
                return result[0]
        except Exception as e:
            logger.warning(f"Failed to encode query: {e}")
        return None

    def _video_filter(self, alias: str, video_ids: list[str] | None) -> str:
        if video_ids:
            return f"FILTER {alias}.video_id IN @video_ids"
        return ""

    def _video_bind(self, video_ids: list[str] | None) -> dict[str, Any]:
        if video_ids:
            return {"video_ids": video_ids}
        return {}

    def _user_filter(self, alias: str, user_id: str | None) -> str:
        if user_id:
            return f"FILTER {alias}.user_id == @user_id"
        return ""

    def _user_bind(self, user_id: str | None) -> dict[str, Any]:
        if user_id:
            return {"user_id": user_id}
        return {}

    def _execute_aql(self, aql: str, bind_vars: dict[str, Any]) -> list[dict[str, Any]]:
        return list(self.db.aql.execute(aql, bind_vars=bind_vars)) #type:ignore

    def _get_effective_context(
        self,
        user_id: str | None,
        video_ids: list[str] | None,
    ) -> tuple[str | None, list[str] | None]:
        effective_user_id = user_id or self._user_id
        effective_video_ids = video_ids or self._video_ids
        return effective_user_id, effective_video_ids

    def _store_result(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> ToolResult:
        handle_id = hashlib.md5(
            json.dumps({"tool": tool_name, "args": kwargs}, sort_keys=True).encode()
        ).hexdigest()[:8]

        self._result_store[handle_id] = results

        content = (
            f"Found {len(results)} result(s).\n"
            f"Handle ID: {handle_id}\n"
            f"To view details: use `view_kg_result(handle_id='{handle_id}', view_mode='detailed')`\n\n"
            f"{self._get_detailed(results, 10)}"
        )

        return ToolResult(content=content)

    def _get_brief(self, results: list[dict[str, Any]], top_n: int = 5) -> str:
        sorted_results = sorted(
            results, key=lambda x: x.get("score", x.get("rrf_score", 0)), reverse=True
        )[:top_n]

        lines = [
            f"Total: {len(results)} result(s)",
            f"Top {min(top_n, len(sorted_results))}:",
        ]

        for i, item in enumerate(sorted_results):
            name = item.get("entity_name") or item.get("caption") or item.get("title") or item.get("text") or item.get("_key", "unknown")
            score = item.get("score") or item.get("rrf_score") or item.get("combined_score", 0)
            video_id = item.get("video_id", "unknown")
            lines.append(f"  {i}. [{video_id[:8]}...] {name[:50]} (score: {score:.3f})")

        return "\n".join(lines)

    def _get_detailed(self, results: list[dict[str, Any]], top_n: int = 5) -> str:
        """Get detailed representation of top N results."""
        sorted_results = sorted(
            results, key=lambda x: x.get("score", x.get("rrf_score", 0)), reverse=True
        )[:top_n]

        lines = [
            "=== Detailed KG Results ===",
            f"Total: {len(results)} result(s)",
            f"Top {min(top_n, len(sorted_results))}:",
            "",
        ]

        for i, item in enumerate(sorted_results):
            lines.append(f"[{i}] {json.dumps(item, indent=2, default=str)}")
            lines.append("")

        return "\n".join(lines)

    @tool(
        description=(
            "Search for entities (people, objects, locations, concepts) in the knowledge graph using semantic similarity. "
            "Returns entities ranked by cosine similarity to the query embedding.\n\n"
            "Typical workflow - Entity discovery and exploration:\n"
            "  1. This tool - find entities matching your query\n"
            "  2. view_kg_result - inspect detailed results with handle_id\n"
            "  3. traverse_from_entity - explore relationships from found entities\n"
            "When to use:\n"
            "  - Finding specific people, objects, or locations in videos\n"
            "  - Discovering entities related to a concept or theme\n"
            "  - Starting point for graph exploration\n\n"
            "Related tools:\n"
            "  - traverse_from_entity: Explore relationships from found entities\n"
            "  - view_kg_result: Inspect cached results\n"
            "  - search_events: Find events/entities in entities\n"
            "  - triple_hybrid_search: More powerful multi-method search\n"
            "  - search.get_images_from_qwenvl_query: Find visual moments with entity\n\n"
            "Args:\n"
            "  query (str): Search query text (REQUIRED)\n"
            "  top_k (int): Number of results to return (default 10)\n"
            "  video_ids (list[str] | None): Optional list of video IDs to filter\n"
            "  user_id (str | None): Optional user ID to filter\n"
            "  min_score (float): Minimum similarity score threshold (default 0.5)"
        ),
        instructions=(
            "Use this when you need to find specific people, objects, or locations mentioned in videos.\n\n"
            "Best paired with: traverse_from_entity (explore relationships), view_kg_result (inspect results). "
            "Follow up with: view_kg_result using handle_id, then traverse_from_entity with entity_key. "
            "Use min_score to filter out low-quality matches (default 0.5)."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def search_entities_semantic(
        self,
        query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        min_score: float = 0.5,
    ) -> ToolResult:
        effective_user_id, effective_video_ids = self._get_effective_context(user_id, video_ids)

        logger.info(f"{effective_user_id=}, {effective_video_ids=}")
        kwargs = {
            "query": query,
            "top_k": top_k,
            "video_ids": effective_video_ids,
            "user_id": effective_user_id,
            "min_score": min_score,
        }

        try:
            query_emb = await self._encode_query_async(query)
            if query_emb is None:
                return ToolResult(content="Error: MMBert client required for semantic search")

            vf = self._video_filter("e", effective_video_ids)
            uf = self._user_filter("e", effective_user_id)

            aql = f"""
            FOR e IN entities
                {uf}
                {vf}
                LET score = COSINE_SIMILARITY(e.semantic_embedding, @query)
                FILTER score > @min_score
                SORT score DESC
                LIMIT @top_k
                RETURN {{
                    _key: e._key,
                    video_id: e.video_id,
                    user_id: e.user_id,
                    entity_name: e.entity_name,
                    entity_type: e.entity_type,
                    description: e["desc"],
                    first_seen_segment: e.first_seen_segment,
                    last_seen_segment: e.last_seen_segment,
                    score: score
                }}
            """

            bind: dict[str, Any] = {
                "query": query_emb,
                "top_k": top_k,
                "min_score": min_score,
                **self._video_bind(effective_video_ids),
                **self._user_bind(effective_user_id),
            }
           

            results = self._execute_aql(aql, bind)
            return self._store_result("search_entities_semantic", kwargs, results)

        except Exception as e:
            logger.error(f"[KGSearchToolkit] search_entities_semantic failed: {e}")
            return ToolResult(content=f"Error: Entity search failed - {str(e)}")

    @tool(
        description=(
            "Search for events (segment-level actions/scenes) in the knowledge graph using semantic similarity. "
            "Events represent meaningful segments of video content with temporal bounds.\n\n"
            "Typical workflow - Event discovery:\n"
            "  1. This tool - find events matching your query\n"
            "  2. view_kg_result - inspect detailed results with handle_id\n"
            "  3. search_entities_semantic - find entities involved in events\n"
            "  4. utility.get_related_asr_from_segment - get spoken context\n\n"
            "When to use:\n"
            "  - Finding specific scenes or actions that occurred in videos\n"
            "  - Need time-bounded information about what happened\n"
            "  - Looking for events with temporal information (start_time, end_time)\n\n"
            "Related tools:\n"
            "  - search_entities_semantic: Find entities involved in events\n"
            "  - search_micro_events: For more granular frame-level details\n"
            "  - view_kg_result: Inspect cached results\n"
            "  - utility.get_related_asr_from_segment: Get ASR context\n"
            "  - search.get_segments_from_event_query_mmbert: Find segment for event\n\n"
            "Args:\n"
            "  query (str): Search query text (REQUIRED)\n"
            "  top_k (int): Number of results to return (default 10)\n"
            "  video_ids (list[str] | None): Optional list of video IDs to filter\n"
            "  user_id (str | None): Optional user ID to filter\n"
            "  min_score (float): Minimum similarity score threshold (default 0.5)"
        ),
        instructions=(
            "Use this to find specific scenes or actions that occurred in videos.\n\n"
            "Events include temporal information (start_time, end_time) and captions describing the action. "
            "Best paired with: search_entities_semantic (find involved entities), view_kg_result (inspect results). "
            "Follow up with: view_kg_result using handle_id. "
            "Use search_micro_events for more granular frame-level details."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def search_events(
        self,
        query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        min_score: float = 0.5,
    ) -> ToolResult:
        effective_user_id, effective_video_ids = self._get_effective_context(user_id, video_ids)

        kwargs = {
            "query": query,
            "top_k": top_k,
            "video_ids": effective_video_ids,
            "user_id": effective_user_id,
            "min_score": min_score,
        }

        try:
            query_emb = await self._encode_query_async(query)
            if query_emb is None:
                return ToolResult(content="Error: MMBert client required for semantic search")

            vf = self._video_filter("ev", effective_video_ids)
            uf = self._user_filter("ev", effective_user_id)

            aql = f"""
            FOR ev IN events
                {uf}
                {vf}
                LET score = COSINE_SIMILARITY(ev.semantic_embedding, @query)
                FILTER score > @min_score
                SORT score DESC
                LIMIT @top_k
                RETURN {{
                    _key: ev._key,
                    video_id: ev.video_id,
                    user_id: ev.user_id,
                    segment_index: ev.segment_index,
                    start_time: ev.start_time,
                    end_time: ev.end_time,
                    caption: ev.caption,
                    entities_global: ev.entities_global,
                    score: score
                }}
            """

            results = self._execute_aql(
                aql,
                {
                    "query": query_emb,
                    "top_k": top_k,
                    "min_score": min_score,
                    **self._video_bind(effective_video_ids),
                    **self._user_bind(effective_user_id),
                },
            )
            return self._store_result("search_events", kwargs, results)

        except Exception as e:
            logger.error(f"[KGSearchToolkit] search_events failed: {e}")
            return ToolResult(content=f"Error: Event search failed - {str(e)}")

    @tool(
        description=(
            "Search for micro-events (frame-level granular actions) in the knowledge graph using semantic similarity. "
            "Micro-events are fine-grained descriptions of individual moments within larger events.\n\n"
            "Typical workflow - Fine-grained event search:\n"
            "  1. search_events - find broader events first\n"
            "  2. This tool - get frame-level details within events\n"
            "  3. view_kg_result - inspect detailed results\n"
            "When to use:\n"
            "  - Need precise, frame-level video content retrieval\n"
            "  - Micro-events contain detailed text descriptions of specific moments\n"
            "  - Each micro-event has a parent_event and can be linked to entities\n\n"
            "Related tools:\n"
            "  - search_events: Coarser event-level search (start here)\n"
            "  - view_kg_result: Inspect cached results\n"
            "  - search_entities_semantic: Find entities in micro-events\n"
            "  - search.get_images_from_qwenvl_query: Find frame for micro-event\n\n"
            "Args:\n"
            "  query (str): Search query text (REQUIRED)\n"
            "  top_k (int): Number of results to return (default 10)\n"
            "  video_ids (list[str] | None): Optional list of video IDs to filter\n"
            "  user_id (str | None): Optional user ID to filter\n"
            "  min_score (float): Minimum similarity score threshold (default 0.5)"
        ),
        instructions=(
            "Use this for precise, frame-level video content retrieval.\n\n"
            "Micro-events contain detailed text descriptions of specific moments. "
            "Best paired with: search_events (start broader), view_kg_result (inspect results). "
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def search_micro_events(
        self,
        query: str,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        min_score: float = 0.5,
    ) -> ToolResult:
        kwargs = {
            "query": query,
            "top_k": top_k,
            "video_ids": video_ids,
            "user_id": user_id,
            "min_score": min_score,
        }

        try:
            query_emb = await self._encode_query_async(query)
            if query_emb is None:
                return ToolResult(content="Error: MMBert client required for semantic search")

            vf = self._video_filter("m", video_ids)
            uf = self._user_filter("m", user_id)

            aql = f"""
            FOR m IN micro_events
                {uf}
                {vf}
                LET score = COSINE_SIMILARITY(m.semantic_embedding, @query)
                FILTER score > @min_score
                SORT score DESC
                LIMIT @top_k
                RETURN {{
                    _key: m._key,
                    video_id: m.video_id,
                    user_id: m.user_id,
                    text: m.text,
                    segment_index: m.segment_index,
                    start_time: m.start_time,
                    end_time: m.end_time,
                    parent_event_key: m.parent_event_key,
                    entities_global: m.entities_global,
                    score: score
                }}
            """

            results = self._execute_aql(
                aql,
                {
                    "query": query_emb,
                    "top_k": top_k,
                    "min_score": min_score,
                    **self._video_bind(video_ids),
                    **self._user_bind(user_id),
                },
            )
            return self._store_result("search_micro_events", kwargs, results)

        except Exception as e:
            logger.error(f"[KGSearchToolkit] search_micro_events failed: {e}")
            return ToolResult(content=f"Error: Micro-event search failed - {str(e)}")

    @tool(
        description=(
            "Search for communities (clusters of related entities) in the knowledge graph using semantic similarity. "
            "Communities represent thematic groupings of entities that share common context.\n\n"
            "Typical workflow - High-level theme discovery:\n"
            "  1. This tool - find thematic communities\n"
            "  2. view_kg_result - inspect community summaries\n"
            "  3. search_entities_semantic - drill down into entities in communities\n"
            "  4. traverse_from_entity - explore entity relationships\n\n"
            "When to use:\n"
            "  - Understanding overarching themes in video content\n"
            "  - High-level understanding of what topics a video covers\n"
            "  - Finding thematic summaries and groupings of related entities\n\n"
            "Related tools:\n"
            "  - search_entities_semantic: Drill down into entities\n"
            "  - view_kg_result: Inspect cached results\n"
            "  - search_events: Find events in communities\n\n"
            "Args:\n"
            "  query (str): Search query text (REQUIRED)\n"
            "  top_k (int): Number of results to return (default 5)\n"
            "  video_ids (list[str] | None): Optional list of video IDs to filter\n"
            "  user_id (str | None): Optional user ID to filter\n"
            "  min_score (float): Minimum similarity score threshold (default 0.4)"
        ),
        instructions=(
            "Use this to find thematic summaries and groupings of related entities.\n\n"
            "Each community has a title, summary, and member entities. "
            "Best paired with: search_entities_semantic (drill down), view_kg_result (inspect results). "
            "Follow up with: view_kg_result using handle_id, then search_entities_semantic to find entities in the community."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def search_communities(
        self,
        query: str,
        top_k: int = 5,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        min_score: float = 0.4,
    ) -> ToolResult:
        kwargs = {
            "query": query,
            "top_k": top_k,
            "video_ids": video_ids,
            "user_id": user_id,
            "min_score": min_score,
        }

        try:
            query_emb = await self._encode_query_async(query)
            if query_emb is None:
                return ToolResult(content="Error: MMBert client required for semantic search")

            vf = self._video_filter("c", video_ids)
            uf = self._user_filter("c", user_id)

            aql = f"""
            FOR c IN communities
                {uf}
                {vf}
                LET score = COSINE_SIMILARITY(c.semantic_embedding, @query)
                FILTER score > @min_score
                SORT score DESC
                LIMIT @top_k
                RETURN {{
                    _key: c._key,
                    video_id: c.video_id,
                    user_id: c.user_id,
                    title: c.title,
                    summary: c.summary,
                    size: c.size,
                    score: score
                }}
            """

            results = self._execute_aql(
                aql,
                {
                    "query": query_emb,
                    "top_k": top_k,
                    "min_score": min_score,
                    **self._video_bind(video_ids),
                    **self._user_bind(user_id),
                },
            )
            return self._store_result("search_communities", kwargs, results)

        except Exception as e:
            logger.error(f"[KGSearchToolkit] search_communities failed: {e}")
            return ToolResult(content=f"Error: Community search failed - {str(e)}")

    @tool(
        description=(
            "Traverse the knowledge graph from a seed entity to discover connected nodes (entities, events, micro-events, communities). "
            "Explores relationships and finds related content through graph edges.\n\n"
            "Typical workflow - Graph exploration:\n"
            "  1. search_entities_semantic - find a seed entity of interest\n"
            "  2. This tool - traverse from the entity to discover relationships\n"
            "  3. view_kg_result - inspect connected nodes\n"
            "When to use:\n"
            "  - After finding an entity of interest, discover related content\n"
            "  - Understanding how entities relate to events and other entities\n"
            "  - Exploring the knowledge graph structure\n\n"
            "Traversal depth:\n"
            "  - depth=1: Direct connections (immediate neighbors)\n"
            "  - depth=2: Two hops (extended network)\n"
            "  - Higher depths may return many results\n\n"
            "Related tools:\n"
            "  - search_entities_semantic: Find seed entities (use before this)\n"
            "  - view_kg_result: Inspect traversal results\n"
            "  - triple_hybrid_search: Alternative multi-method search\n\n"
            "Args:\n"
            "  entity_key (str): The _key of the seed entity (REQUIRED)\n"
            "  max_depth (int): Maximum traversal depth (default 2)\n"
            "  video_ids (list[str] | None): Optional list of video IDs to filter results\n"
            "  user_id (str | None): Optional user ID to filter results"
        ),
        instructions=(
            "Use this after finding an entity of interest to discover related content.\n\n"
            "Best paired with: search_entities_semantic (find seed entity first), view_kg_result (inspect results). "
            "Follow up with: view_kg_result using handle_id. "
            "Set max_depth to control how far to traverse (default 2 hops)."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    def traverse_from_entity(
        self,
        entity_key: str,
        max_depth: int = 2,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
    ) -> ToolResult:
        kwargs = {
            "entity_key": entity_key,
            "max_depth": max_depth,
            "video_ids": video_ids,
            "user_id": user_id,
        }

        try:
            vf = self._video_filter("v", video_ids)
            uf = self._user_filter("v", user_id)

            aql = f"""
            WITH entities, events, micro_events, communities
            FOR v, e, p IN 1..@depth ANY @start
            GRAPH @graph_name
            OPTIONS {{ uniqueVertices: "global", order: "bfs" }}
            {uf}
            {vf}
            RETURN {{
                _key: v._key,
                _id: v._id,
                video_id: v.video_id,
                user_id: v.user_id,
                label: v.entity_name || v.caption || v.title || v.text || "",
                edge_type: e.relation_type || e.edge_type || "link",
                weight: e.weight,
                path_length: LENGTH(p.vertices)
            }}
            """

            results = self._execute_aql(
                aql,
                {
                    "start": f"entities/{entity_key}",
                    "depth": max_depth,
                    "graph_name": self.graph_name,
                    **self._video_bind(video_ids),
                    **self._user_bind(user_id),
                },
            )
            return self._store_result("traverse_from_entity", kwargs, results)

        except Exception as e:
            logger.error(f"[KGSearchToolkit] traverse_from_entity failed: {e}")
            return ToolResult(content=f"Error: Graph traversal failed - {str(e)}")

    @tool(
        description=(
            "A 'Broad Discovery' tool. Performs parallel semantic (vector) search across three distinct layers: "
            "Entities (people/objects), Events (long-term actions), and Micro-events (frame-level actions). "
            "Optimized for speed and diversity of context.\n\n"
            "Typical workflow - Broad discovery:\n"
            "  1. This tool - broad search across all layers (FIRST PASS)\n"
            "  2. view_kg_result - inspect results using handle_id\n"
            "  3. search_entities_semantic/search_events - drill down into specific layers\n"
            "  4. traverse_from_entity - explore relationships from found entities\n\n"
            "When to use:\n"
            "  - User's intent is broad or exploratory (e.g., What's happening in this video?)\n"
            "  - First pass to identify relevant content before diving deeper\n"
            "  - General questions about video content\n\n"
            "Related tools:\n"
            "  - search_entities_semantic: Entity-only search\n"
            "  - search_events: Event-only search\n"
            "  - search_micro_events: Micro-event-only search\n"
            "  - view_kg_result: Inspect cached results\n\n"
            "Args:\n"
            "  query (str): Search query text (REQUIRED)\n"
            "  video_ids (list[str] | None): Optional list of video IDs to filter\n"
            "  user_id (str | None): Optional user ID to filter\n"
            "  min_score (float): Minimum similarity score threshold (default 0.4)\n"
            "  top_k (int): Number of results per collection (default 5)"
        ),
        instructions=(
            "Use this when the user's intent is broad or exploratory.\n\n"
            "Use this as the 'First Pass' tool to identify which video segments or objects are relevant before diving deeper. "
            "Best paired with: view_kg_result (inspect results), search_entities_semantic/search_events (drill down). "
            "Follow up with: view_kg_result using handle_id, then specific layer search tools."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def multi_granularity_search(
        self,
        query: str,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        min_score: float = 0.4,
        top_k: int = 5,
    ) -> ToolResult:
        kwargs = {
            "query": query,
            "video_ids": video_ids,
            "user_id": user_id,
            "top_k": top_k,
        }

        try:
            query_emb = await self._encode_query_async(query)
            if query_emb is None:
                return ToolResult(content="Error: MMBert client required for semantic search")

            video_bind = self._video_bind(video_ids)
            user_bind = self._user_bind(user_id)
            all_results: list[dict[str, Any]] = []

            collections = [
                ("entities", "semantic_embedding"),
                ("events", "semantic_embedding"),
                ("micro_events", "semantic_embedding"),
            ]

            for coll, emb_field in collections:
                vf = self._video_filter("doc", video_ids)
                uf = self._user_filter("doc", user_id)
                aql = f"""
                FOR doc IN {coll}
                {uf}
                {vf}
                LET score = COSINE_SIMILARITY(doc.{emb_field}, @vec)
                FILTER score > {min_score}
                SORT score DESC LIMIT @top_k
                RETURN MERGE(
                    UNSET(doc, "semantic_embedding", "structural_embedding_entity_only", "structural_embedding_entity_event", "structural_embedding_full"),
                    {{score: score, collection: '{coll}'}}
                )
                """
                results = self._execute_aql(aql, {"vec": query_emb, "top_k": top_k, **video_bind, **user_bind})
                all_results.extend(results)

            lines = [
                f"=== Multi-Granularity Search Results ===",
                f"Query: {query}",
                f"Total results: {len(all_results)}",
                "",
            ]

            for coll in ["entities", "events", "micro_events"]:
                coll_results = [r for r in all_results if r.get("collection") == coll]
                if coll_results:
                    lines.append(f"--- {coll.upper()} ({len(coll_results)}) ---")
                    for item in coll_results:
                        name = item.get("entity_name") or item.get("caption") or item.get("text", "N/A")
                        score = item.get("score", 0)
                        lines.append(f"  - {name} (score: {score:.3f})")
                    lines.append("")

            self._result_store[hashlib.md5(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()[:8]] = all_results
            return ToolResult(content="\n".join(lines))

        except Exception as e:
            logger.error(f"[KGSearchToolkit] multi_granularity_search failed: {e}")
            return ToolResult(content=f"Error: Multi-granularity search failed - {str(e)}")

    @tool(
        description=(
            "Keyword-based BM25 full-text search across knowledge graph collections. "
            "Uses ArangoSearch with inverted indexes for fast keyword and phrase matching.\n\n"
            "Typical workflow - Keyword search:\n"
            "  1. This tool - find exact keyword matches\n"
            "  2. view_kg_result - inspect results using handle_id\n"
            "  3. traverse_from_entity - explore relationships from found entities\n"
            "  4. Or triple_hybrid_search - for more comprehensive search\n\n"
            "When to use:\n"
            "  - Exact keyword matching when semantic search is not needed\n"
            "  - Finding specific names, technical terms, or exact phrases\n"
            "  - Fast keyword search across all KG collections\n\n"
            "Related tools:\n"
            "  - triple_hybrid_search: More powerful search combining BM25 + semantic + graph\n"
            "  - search_entities_semantic: Semantic entity search\n"
            "  - view_kg_result: Inspect cached results\n"
            "  - ocr.search_ocr_text: For text visible in frames\n\n"
            "Args:\n"
            "  query (str): Search query text - keywords/phrases (REQUIRED)\n"
            "  collections (list[str] | None): Optional list of collections to search. "
            "Valid values: ['entities', 'events', 'micro_events', 'communities']. Default: all\n"
            "  top_k (int): Number of results per collection (default 10)\n"
            "  video_ids (list[str] | None): Optional list of video IDs to filter\n"
            "  user_id (str | None): Optional user ID to filter\n"
            "  min_score (float): Minimum BM25 score threshold (default 0.1)"
        ),
        instructions=(
            "Use this for exact keyword matching when semantic search is not needed.\n\n"
            "Best paired with: view_kg_result (inspect results), triple_hybrid_search (more powerful alternative). "
            "Follow up with: view_kg_result using handle_id. "
            "Use the 'collections' parameter to limit which collections to search."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def search_bm25(
        self,
        query: str,
        collections: list[str] | None = None,
        top_k: int = 10,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        min_score: float = 0.1,
    ) -> ToolResult:
        kwargs = {
            "query": query,
            "collections": collections,
            "top_k": top_k,
            "video_ids": video_ids,
            "user_id": user_id,
            "min_score": min_score,
        }

        valid_collections = ["entities", "events", "micro_events", "communities"]
        search_collections = collections if collections else valid_collections
        search_collections = [c for c in search_collections if c in valid_collections]

        if not search_collections:
            return ToolResult(content="Error: No valid collections specified")

        try:
            vf = self._video_filter("doc", video_ids)
            uf = self._user_filter("doc", user_id)
            video_bind = self._video_bind(video_ids)
            user_bind = self._user_bind(user_id)

            all_results: list[dict[str, Any]] = []

            for coll in search_collections:
                # Build collection-specific SEARCH clause
                if coll == "entities":
                    search_fields = 'doc.entity_name IN TOKENS(@text, "text_en") OR doc["desc"] IN TOKENS(@text, "text_en")'
                    return_fields = """
                        _key: doc._key,
                        video_id: doc.video_id,
                        user_id: doc.user_id,
                        entity_name: doc.entity_name,
                        entity_type: doc.entity_type,
                        description: doc["desc"],
                        score: s,
                        collection: 'entities'
                    """
                elif coll == "events":
                    search_fields = 'doc.caption IN TOKENS(@text, "text_en")'
                    return_fields = """
                        _key: doc._key,
                        video_id: doc.video_id,
                        user_id: doc.user_id,
                        segment_index: doc.segment_index,
                        caption: doc.caption,
                        start_time: doc.start_time,
                        end_time: doc.end_time,
                        score: s,
                        collection: 'events'
                    """
                elif coll == "micro_events":
                    search_fields = 'doc.text IN TOKENS(@text, "text_en") OR doc.related_caption_context IN TOKENS(@text, "text_en")'
                    return_fields = """
                        _key: doc._key,
                        video_id: doc.video_id,
                        user_id: doc.user_id,
                        text: doc.text,
                        segment_index: doc.segment_index,
                        micro_index: doc.micro_index,
                        start_time: doc.start_time,
                        end_time: doc.end_time,
                        score: s,
                        collection: 'micro_events'
                    """
                else:  # communities
                    search_fields = 'doc.title IN TOKENS(@text, "text_en") OR doc.summary IN TOKENS(@text, "text_en")'
                    return_fields = """
                        _key: doc._key,
                        video_id: doc.video_id,
                        user_id: doc.user_id,
                        title: doc.title,
                        summary: doc.summary,
                        size: doc.size,
                        score: s,
                        collection: 'communities'
                    """

                aql = f"""
                FOR doc IN {self.search_view}
                SEARCH ANALYZER({search_fields}, 'text_en')
                OPTIONS {{ parallelism: 4 }}
                FILTER IS_SAME_COLLECTION('{coll}', doc)
                {uf}
                {vf}
                LET s = BM25(doc)
                FILTER s > @min_score
                SORT s DESC
                LIMIT @top_k
                RETURN {{{return_fields}}}
                """

                results = self._execute_aql(
                    aql,
                    {"text": query, "min_score": min_score, "top_k": top_k, **video_bind, **user_bind},
                )
                all_results.extend(results)

            # Sort all results by score
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

            # Build summary
            lines = [
                f"=== BM25 Full-Text Search Results ===",
                f"Query: '{query}'",
                f"Collections: {search_collections}",
                f"Total results: {len(all_results)}",
                "",
            ]

            for coll in search_collections:
                coll_results = [r for r in all_results if r.get("collection") == coll]
                if coll_results:
                    lines.append(f"--- {coll.upper()} ({len(coll_results)}) ---")
                    for item in coll_results[:3]:
                        name = item.get("entity_name") or item.get("caption") or item.get("text") or item.get("title", "N/A")
                        score = item.get("score", 0)
                        lines.append(f"  - {name[:60]} (BM25: {score:.3f})")
                    lines.append("")

            handle_id = hashlib.md5(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()[:8]
            self._result_store[handle_id] = all_results
            lines.append(f"Handle ID: {handle_id}")

            return ToolResult(content="\n".join(lines))

        except Exception as e:
            logger.error(f"[KGSearchToolkit] search_bm25 failed: {e}")
            return ToolResult(content=f"Error: BM25 search failed - {str(e)}")

    async def _triple_hybrid_search_impl(
        self,
        query: str,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        top_k: int = 10,
        seed_entities: list[str] | None = None,
        search_all_collections: bool = False,
    ) -> list[dict[str, Any]]:
        query_emb = await self._encode_query_async(query)
        if query_emb is None:
            raise ValueError("MMBert client required for semantic search")

        vf = self._video_filter("doc", video_ids)
        uf = self._user_filter("doc", user_id)
        video_bind = self._video_bind(video_ids)
        user_bind = self._user_bind(user_id)
        seed_bind = {"seeds": seed_entities} if seed_entities else {"seeds": []}

        if search_all_collections:
            bm25_entities = f"""
                FOR doc IN {self.search_view}
                SEARCH ANALYZER(
                    doc.entity_name IN TOKENS(@text, 'text_en') OR
                    doc["desc"] IN TOKENS(@text, 'text_en'),
                    'text_en'
                )
                OPTIONS {{ parallelism: 4 }}
                FILTER IS_SAME_COLLECTION('entities', doc)
                {uf}
                {vf}
                LET s = BM25(doc)
                SORT s DESC LIMIT 50
                RETURN {{_key: doc._key, score: s, source: 'bm25', collection: 'entities'}}
                """

            bm25_events = f"""
                FOR doc IN {self.search_view}
                SEARCH ANALYZER(doc.caption IN TOKENS(@text, 'text_en'), 'text_en')
                OPTIONS {{ parallelism: 4 }}
                FILTER IS_SAME_COLLECTION('events', doc)
                {uf}
                {vf}
                LET s = BM25(doc)
                SORT s DESC LIMIT 50
                RETURN {{_key: doc._key, score: s, source: 'bm25', collection: 'events'}}
                """

            bm25_micro = f"""
                FOR doc IN {self.search_view}
                SEARCH ANALYZER(
                    doc.text IN TOKENS(@text, 'text_en') OR
                    doc.related_caption_context IN TOKENS(@text, 'text_en'),
                    'text_en'
                )
                OPTIONS {{ parallelism: 4 }}
                FILTER IS_SAME_COLLECTION('micro_events', doc)
                {uf}
                {vf}
                LET s = BM25(doc)
                SORT s DESC LIMIT 50
                RETURN {{_key: doc._key, score: s, source: 'bm25', collection: 'micro_events'}}
                """

            bm25_communities = f"""
                FOR doc IN {self.search_view}
                SEARCH ANALYZER(
                    doc.title IN TOKENS(@text, 'text_en') OR
                    doc.summary IN TOKENS(@text, 'text_en'),
                    'text_en'
                )
                OPTIONS {{ parallelism: 4 }}
                FILTER IS_SAME_COLLECTION('communities', doc)
                {uf}
                {vf}
                LET s = BM25(doc)
                SORT s DESC LIMIT 50
                RETURN {{_key: doc._key, score: s, source: 'bm25', collection: 'communities'}}
                """

            bm25_hits = []
            for bm25_aql in [bm25_entities, bm25_events, bm25_micro, bm25_communities]:
                bm25_hits.extend(self._execute_aql(bm25_aql, {"text": query, **video_bind, **user_bind}))

            vec_entities = f"""
                FOR doc IN entities
                {uf}
                {vf}
                LET s = COSINE_SIMILARITY(doc.semantic_embedding, @vec)
                FILTER s > 0.4
                SORT s DESC LIMIT 50
                RETURN {{_key: doc._key, score: s, source: 'vector', collection: 'entities'}}
                """

            vec_events = f"""
                FOR doc IN events
                {uf}
                {vf}
                LET s = COSINE_SIMILARITY(doc.semantic_embedding, @vec)
                FILTER s > 0.4
                SORT s DESC LIMIT 50
                RETURN {{_key: doc._key, score: s, source: 'vector', collection: 'events'}}
                """

            vec_micro = f"""
                FOR doc IN micro_events
                {uf}
                {vf}
                LET s = COSINE_SIMILARITY(doc.semantic_embedding, @vec)
                FILTER s > 0.4
                SORT s DESC LIMIT 50
                RETURN {{_key: doc._key, score: s, source: 'vector', collection: 'micro_events'}}
                """

            vec_communities = f"""
                FOR doc IN communities
                {uf}
                {vf}
                LET s = COSINE_SIMILARITY(doc.semantic_embedding, @vec)
                FILTER s > 0.4
                SORT s DESC LIMIT 50
                RETURN {{_key: doc._key, score: s, source: 'vector', collection: 'communities'}}
                """

            vec_hits = []
            for vec_aql in [vec_entities, vec_events, vec_micro, vec_communities]:
                vec_hits.extend(self._execute_aql(vec_aql, {"vec": query_emb, **video_bind, **user_bind}))

        else:


            aql = f"""
            WITH entities, {self.search_view}
            LET bm25_hits = (
                FOR doc IN {self.search_view}
                
                
                SEARCH ANALYZER(
                    doc.entity_name IN TOKENS(@text, 'text_en') OR
                    doc["desc"] IN TOKENS(@text, 'text_en'),
                    'text_en'
                )
                OPTIONS {{ parallelism: 4 }}
                FILTER IS_SAME_COLLECTION('entities', doc)
                {uf}
                {vf}

                LET s = BM25(doc)
                SORT s DESC LIMIT 50
                RETURN {{_key: doc._key, score: s, source: 'bm25'}}
            )

            LET vec_hits = (
                FOR doc IN entities
                {uf}
                {vf}
                LET s = COSINE_SIMILARITY(doc.semantic_embedding, @vec)
                FILTER s > 0.4
                SORT s DESC LIMIT 50
                RETURN {{_key: doc._key, score: s, source: 'vector'}}
            )

            LET graph_hits = (
                FOR seed IN @seeds
                FOR v, e, p IN 1..2 ANY CONCAT('entities/', seed)
                GRAPH @graph_name
                OPTIONS {{ parallelism: 4 }}
                FILTER IS_SAME_COLLECTION('entities', v)
                {uf}
                LET s = COSINE_SIMILARITY(v.semantic_embedding, @vec)
                SORT s DESC LIMIT 50
                RETURN DISTINCT {{_key: v._key, score: s, source: 'graph'}}
            )

            LET all_keys = UNIQUE(APPEND(bm25_hits[*]._key, vec_hits[*]._key, graph_hits[*]._key))

            FOR k IN all_keys
                LET vrank = POSITION(vec_hits[*]._key, k, true)
                LET trank = POSITION(bm25_hits[*]._key, k, true)
                LET grank = POSITION(graph_hits[*]._key, k, true)

                LET rrf =
                    1.0 / (60 + (vrank >= 0 ? vrank + 1 : 101)) +
                    1.0 / (60 + (trank >= 0 ? trank + 1 : 101)) +
                    1.0 / (60 + (grank >= 0 ? grank + 1 : 101))

                SORT rrf DESC LIMIT @top_k
                LET doc = DOCUMENT('entities', k)
                RETURN MERGE(KEEP(doc, "_key", "_id", "video_id", "user_id", "global_entity_id", "entity_name", "entity_type", "desc", "first_seen_segment", "last_seen_segment"), {{rrf_score: rrf}})
            """

            results = self._execute_aql(
                aql,
                {
                    "text": query,
                    "vec": query_emb,
                    "top_k": top_k,
                    "graph_name": self.graph_name,
                    **video_bind,
                    **user_bind,
                    **seed_bind,
                },
            )
            return results

        graph_hits = []
        if seed_entities:
            graph_aql = f"""
            FOR seed IN @seeds
            FOR v, e, p IN 1..2 ANY CONCAT('entities/', seed)
            GRAPH @graph_name
            OPTIONS {{ parallelism: 4 }}
            FILTER IS_SAME_COLLECTION('entities', v)
            {uf}
            LET s = COSINE_SIMILARITY(v.semantic_embedding, @vec)
            SORT s DESC LIMIT 50
            RETURN DISTINCT {{_key: v._key, score: s, source: 'graph', collection: 'entities'}}
            """
            graph_hits = self._execute_aql(
                graph_aql,
                {"seeds": seed_entities, "vec": query_emb, "graph_name": self.graph_name, **video_bind, **user_bind},
            )

        all_hits = bm25_hits + vec_hits + graph_hits

        all_keys = []
        seen = set()
        for hit in all_hits:
            coll = hit.get("collection", "entities")
            key = hit.get("_key")
            composite_key = f"{coll}/{key}"
            if composite_key not in seen:
                seen.add(composite_key)
                all_keys.append((coll, key, hit))

        fused_results = []
        for coll, key, hit in all_keys:
            vrank = next((i for i, h in enumerate(vec_hits) if h["_key"] == key and h.get("collection", "entities") == coll), -1)
            trank = next((i for i, h in enumerate(bm25_hits) if h["_key"] == key and h.get("collection", "entities") == coll), -1)
            grank = next((i for i, h in enumerate(graph_hits) if h["_key"] == key and h.get("collection", "entities") == coll), -1)

            rrf = (
                1.0 / (60 + (vrank + 1 if vrank >= 0 else 101)) +
                1.0 / (60 + (trank + 1 if trank >= 0 else 101)) +
                1.0 / (60 + (grank + 1 if grank >= 0 else 101))
            )

            doc_aql = f"""
            FOR d IN {coll}
            FILTER d._key == @key
            LIMIT 1
            RETURN UNSET(d, "semantic_embedding", "structural_embedding_entity_only", "structural_embedding_entity_event", "structural_embedding_full")
            """
            docs = self._execute_aql(doc_aql, {"key": key})
            if docs:
                doc = docs[0]
                doc["rrf_score"] = rrf
                doc["collection"] = coll
                fused_results.append(doc)

        fused_results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        return fused_results[:top_k]

    @tool(
        description=(
            "Triple hybrid search combining BM25 keyword search, semantic vector search, and graph-based retrieval. "
            "Uses Reciprocal Rank Fusion (RRF) to merge and rank results from all three methods. "
            "The most powerful retrieval method for finding relevant content.\n\n"
            "Typical workflow - Comprehensive search:\n"
            "  1. This tool - multi-method search (most powerful)\n"
            "  2. view_kg_result - inspect ranked results\n"
            "  3. traverse_from_entity - explore relationships from top entities\n"
            "When to use:\n"
            "  - Primary retrieval tool for complex queries\n"
            "  - Need comprehensive search combining multiple methods\n"
            "  - Best results for difficult queries\n\n"
            "Methods combined:\n"
            "  1. BM25: Keyword matching for exact terms\n"
            "  2. Vector: Semantic similarity for meaning\n"
            "  3. Graph: Relationship-based expansion from seed entities\n\n"
            "Related tools:\n"
            "  - search_entities_semantic: Simpler semantic-only search\n"
            "  - search_bm25: Simpler keyword-only search\n"
            "  - view_kg_result: Inspect cached results\n"
            "  - retrieve_for_rag: Comprehensive RAG retrieval\n\n"
            "Args:\n"
            "  query (str): Search query text (REQUIRED)\n"
            "  video_ids (list[str] | None): Optional list of video IDs to filter\n"
            "  user_id (str | None): Optional user ID to filter\n"
            "  top_k (int): Number of results to return (default 10)\n"
            "  seed_entities (list[str] | None): Optional entity keys for graph-based expansion\n"
            "  search_all_collections (bool): Search across all collections if True (default False)"
        ),
        instructions=(
            "Use this as the primary retrieval tool for complex queries.\n\n"
            "Combines keyword matching (BM25), semantic understanding (vectors), and graph relationships. "
            "Best paired with: view_kg_result (inspect results), traverse_from_entity (explore relationships). "
            "Follow up with: view_kg_result using handle_id. "
            "Set search_all_collections=True to search across entities, events, micro_events, and communities."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def triple_hybrid_search(
        self,
        query: str,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        top_k: int = 10,
        seed_entities: list[str] | None = None,
        search_all_collections: bool = False,
    ) -> ToolResult:
        kwargs = {
            "query": query,
            "video_ids": video_ids,
            "user_id": user_id,
            "top_k": top_k,
            "seed_entities": seed_entities,
            "search_all_collections": search_all_collections,
        }

        try:
            results = await self._triple_hybrid_search_impl(
                query=query,
                video_ids=video_ids,
                user_id=user_id,
                top_k=top_k,
                seed_entities=seed_entities,
                search_all_collections=search_all_collections,
            )
            return self._store_result("triple_hybrid_search", kwargs, results)

        except Exception as e:
            logger.error(f"[KGSearchToolkit] triple_hybrid_search failed: {e}")
            return ToolResult(content=f"Error: Triple hybrid search failed - {str(e)}")

    @tool(
        description=(
            "Comprehensive RAG retrieval from the knowledge graph combining all retrieval methods. "
            "Retrieves entities, events, micro-events, communities, and graph context for RAG pipelines.\n\n"
            "Typical workflow - RAG context building:\n"
            "  1. This tool - comprehensive retrieval for RAG\n"
            "  2. view_kg_result - inspect retrieved context\n"
            "  3. traverse_from_entity - get additional relationship context\n"
            "  4. utility.get_related_asr_from_segment - supplement with ASR context\n\n"
            "When to use:\n"
            "  - Building context for LLM-based question answering\n"
            "  - Comprehensive retrieval across all graph element types\n"
            "  - Preparing structured context for RAG applications\n\n"
            "Retrieval includes:\n"
            "  - Entities (via triple-hybrid or semantic search)\n"
            "  - Events (semantic search)\n"
            "  - Micro-events (semantic search)\n"
            "  - Communities (semantic search)\n"
            "  - Graph context via traversal\n\n"
            "Related tools:\n"
            "  - triple_hybrid_search: Entity-focused hybrid search\n"
            "  - view_kg_result: Inspect retrieved context\n"
            "  - search_entities_semantic: Entity-only search\n"
            "  - utility.get_related_asr_from_segment: Supplement with ASR\n\n"
            "Args:\n"
            "  query (str): Search query text (REQUIRED)\n"
            "  video_ids (list[str] | None): Optional list of video IDs to filter\n"
            "  user_id (str | None): Optional user ID to filter\n"
            "  top_k_entities (int): Number of entities to retrieve (default 10)\n"
            "  top_k_events (int): Number of events to retrieve (default 5)\n"
            "  top_k_micro (int): Number of micro-events to retrieve (default 5)\n"
            "  top_k_communities (int): Number of communities to retrieve (default 3)\n"
            "  enable_traversal (bool): Enable graph traversal from top entities (default True)\n"
            "  traversal_depth (int): Graph traversal depth (default 1)\n"
            "  use_triple_hybrid (bool): Use triple-hybrid for entity search (default True)"
        ),
        instructions=(
            "Use this when building context for LLM-based question answering.\n\n"
            "Performs comprehensive retrieval across all graph element types. "
            "Best paired with: view_kg_result (inspect context), utility.get_related_asr_from_segment (supplement with ASR). "
            "Follow up with: view_kg_result using handle_id. "
            "Adjust top_k_* parameters to control context size."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def retrieve_for_rag(
        self,
        query: str,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
        top_k_entities: int = 10,
        top_k_events: int = 5,
        top_k_micro: int = 5,
        top_k_communities: int = 3,
        enable_traversal: bool = True,
        traversal_depth: int = 1,
        use_triple_hybrid: bool = True,
    ) -> ToolResult:
        kwargs = {
            "query": query,
            "video_ids": video_ids,
            "user_id": user_id,
            "top_k_entities": top_k_entities,
            "top_k_events": top_k_events,
            "top_k_micro": top_k_micro,
            "top_k_communities": top_k_communities,
            "enable_traversal": enable_traversal,
            "traversal_depth": traversal_depth,
            "use_triple_hybrid": use_triple_hybrid,
        }

        try:
            query_emb = await self._encode_query_async(query)
            if query_emb is None:
                return ToolResult(content="Error: MMBert client required for semantic search")

            video_bind = self._video_bind(video_ids)
            user_bind = self._user_bind(user_id)

            if use_triple_hybrid:
                entities = await self._triple_hybrid_search_impl(
                    query, video_ids=video_ids, user_id=user_id, top_k=top_k_entities
                )
            else:
                vf = self._video_filter("e", video_ids)
                uf = self._user_filter("e", user_id)
                aql = f"""
                FOR e IN entities
                    {uf}
                    {vf}
                    LET score = COSINE_SIMILARITY(e.semantic_embedding, @query)
                    FILTER score > 0.5
                    SORT score DESC
                    LIMIT @top_k
                    RETURN {{
                        _key: e._key,
                        video_id: e.video_id,
                        user_id: e.user_id,
                        entity_name: e.entity_name,
                        entity_type: e.entity_type,
                        description: e["desc"],
                        score: score
                    }}
                """
                entities = self._execute_aql(aql, {"query": query_emb, "top_k": top_k_entities, **video_bind, **user_bind})


            vf = self._video_filter("ev", video_ids)
            uf = self._user_filter("ev", user_id)
            events_aql = f"""
            FOR ev IN events
                {uf}
                {vf}
                LET score = COSINE_SIMILARITY(ev.semantic_embedding, @query)
                FILTER score > 0.5
                SORT score DESC
                LIMIT @top_k
                RETURN {{
                    _key: ev._key,
                    video_id: ev.video_id,
                    user_id: ev.user_id,
                    caption: ev.caption,
                    start_time: ev.start_time,
                    end_time: ev.end_time,
                    score: score
                }}
            """
            events = self._execute_aql(events_aql, {"query": query_emb, "top_k": top_k_events, **video_bind, **user_bind})


            vf = self._video_filter("m", video_ids)
            uf = self._user_filter("m", user_id)
            micro_aql = f"""
            FOR m IN micro_events
                {uf}
                {vf}
                LET score = COSINE_SIMILARITY(m.semantic_embedding, @query)
                FILTER score > 0.5
                SORT score DESC
                LIMIT @top_k
                RETURN {{
                    _key: m._key,
                    video_id: m.video_id,
                    user_id: m.user_id,
                    text: m.text,
                    start_time: m.start_time,
                    end_time: m.end_time,
                    score: score
                }}
            """
            micro_events = self._execute_aql(micro_aql, {"query": query_emb, "top_k": top_k_micro, **video_bind, **user_bind})


            vf = self._video_filter("c", video_ids)
            uf = self._user_filter("c", user_id)
            communities_aql = f"""
            FOR c IN communities
                {uf}
                {vf}
                LET score = COSINE_SIMILARITY(c.semantic_embedding, @query)
                FILTER score > 0.4
                SORT score DESC
                LIMIT @top_k
                RETURN {{
                    _key: c._key,
                    video_id: c.video_id,
                    user_id: c.user_id,
                    title: c.title,
                    summary: c.summary,
                    score: score
                }}
            """
            communities = self._execute_aql(communities_aql, {"query": query_emb, "top_k": top_k_communities, **video_bind, **user_bind})

            # Graph traversal
            graph_context = []
            if enable_traversal and entities:
                for ent in entities[:3]:
                    entity_key = ent.get("_key")
                    if entity_key:
                        vf = self._video_filter("v", video_ids)
                        uf = self._user_filter("v", user_id)
                        traversal_aql = f"""
                        WITH entities, events, micro_events, communities
                        FOR v, e, p IN 1..@depth ANY @start
                        GRAPH @graph_name
                        OPTIONS {{ uniqueVertices: "global", order: "bfs" }}
                        {uf}
                        {vf}
                        RETURN {{
                            _key: v._key,
                            label: v.entity_name || v.caption || v.title || v.text || "",
                            edge_type: e.relation_type || e.edge_type || "link"
                        }}
                        """
                        traversal_results = self._execute_aql(
                            traversal_aql,
                            {
                                "start": f"entities/{entity_key}",
                                "depth": traversal_depth,
                                "graph_name": self.graph_name,
                                **video_bind,
                                **user_bind,
                            },
                        )
                        graph_context.extend(traversal_results)

            seen = set()
            deduped_context = []
            for g in graph_context:
                if g["_key"] not in seen:
                    seen.add(g["_key"])
                    deduped_context.append(g)

            videos_hit: dict[str, int] = {}
            for item in entities + events + micro_events + communities:
                vid = item.get("video_id", "unknown")
                videos_hit[vid] = videos_hit.get(vid, 0) + 1

            lines = [
                "=== RAG Retrieval Results ===",
                f"Query: {query}",
                f"Videos searched: {video_ids or ['all']}",
                "",
                f"Entities: {len(entities)}",
                f"Events: {len(events)}",
                f"Micro-events: {len(micro_events)}",
                f"Communities: {len(communities)}",
                f"Graph context: {len(deduped_context)}",
                "",
                f"Total nodes: {len(entities) + len(events) + len(micro_events) + len(communities) + len(deduped_context)}",
                "",
                "--- Top Entities ---",
            ]

            for ent in entities[:5]:
                lines.append(f"  - {ent.get('entity_name', 'N/A')} ({ent.get('entity_type', 'unknown')}): {ent.get('description', '')[:80]}")

            lines.append("")
            lines.append("--- Top Events ---")
            for ev in events[:3]:
                lines.append(f"  - [{ev.get('start_time', '?')} - {ev.get('end_time', '?')}] {ev.get('caption', 'N/A')[:60]}")

            if communities:
                lines.append("")
                lines.append("--- Top Communities ---")
                for c in communities[:2]:
                    lines.append(f"  - {c.get('title', 'N/A')}: {c.get('summary', '')[:80]}")

            # Store full results
            full_results = {
                "entities": entities,
                "events": events,
                "micro_events": micro_events,
                "communities": communities,
                "graph_context": deduped_context,
                "videos_hit": videos_hit,
            }
            handle_id = hashlib.md5(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()[:8]
            self._result_store[handle_id] = [full_results]  # Wrap in list for consistency

            lines.append("")
            lines.append(f"Handle ID: {handle_id}")

            return ToolResult(content="\n".join(lines))

        except Exception as e:
            logger.error(f"[KGSearchToolkit] retrieve_for_rag failed: {e}")
            return ToolResult(content=f"Error: RAG retrieval failed - {str(e)}")

    @tool(
        description=(
            "View cached knowledge graph search results from previous tool calls. "
            "Retrieve detailed information using the handle_id returned by search tools.\n\n"
            "This tool is the companion to all KG search tools:\n"
            "  - search_entities_semantic\n"
            "  - search_events\n"
            "  - search_micro_events\n"
            "  - search_communities\n"
            "  - traverse_from_entity\n"
            "  - multi_granularity_search\n"
            "  - search_bm25\n"
            "  - triple_hybrid_search\n"
            "  - retrieve_for_rag\n\n"
            "Typical workflow:\n"
            "  1. Call any KG search tool (results are automatically cached)\n"
            "  2. This tool - inspect cached results with different verbosity levels\n"
            "  3. Use entity_key or event details for further exploration\n\n"
            "View modes:\n"
            "  - 'brief': Top N results with essential info (default)\n"
            "  - 'detailed': Top N results with full details\n"
            "  - 'full': All results with full details\n\n"
            "Related tools:\n"
            "  - All KG search tools (they populate this cache)\n"
            "  - traverse_from_entity: Use entity_key from results for exploration\n"
            "  - view_cache_result (Search toolkit): For video/image/audio results\n"
            "  - view_ocr_result (OCR toolkit): For OCR results\n\n"
            "Args:\n"
            "  handle_id (str): Handle ID from previous search (REQUIRED)\n"
            "  view_mode (str): 'brief' (top N summary), 'detailed' (full details), 'full' (all results). Default: 'brief'\n"
            "  top_n (int): Number of results to show in brief/detailed mode. Default: 5\n\n"
            "Note: This tool does NOT accept user_id or video_ids parameters."
        ),
        instructions=(
            "Use this to inspect results from previous searches using the handle_id.\n\n"
            "Best paired with: all KG search tools (they populate this cache). "
            "Follow up with: traverse_from_entity using entity_key from results. "
            "Alternative view tools: view_cache_result (Search), view_ocr_result (OCR)."
        ),
    )
    def view_kg_result(
        self,
        handle_id: str,
        view_mode: Literal["brief", "detailed", "full"] = "brief",
        top_n: int = 5,
    ) -> ToolResult:
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


__all__ = ["KGSearchToolkit"]