"""Index management for ArangoDB Knowledge Graph collections.

Creates and manages vector indexes, inverted indexes, MDI indexes,
and ArangoSearch views for semantic and full-text search.

This module is designed to be called during FastAPI lifespan setup.
"""

from __future__ import annotations

import math
from typing import Any

from loguru import logger


class ArangoIndexManager:
    """Manages ArangoDB indexes for Knowledge Graph collections.

    Creates the following index types:
    - Vector indexes (semantic embeddings: 384-dim, structural embeddings: 128-dim)
    - Inverted indexes (BM25/TF-IDF full-text search)
    - MDI indexes (multi-dimensional interval queries for time ranges)
    - Persistent indexes (fast lookups on video_id, entity_type)
    - ArangoSearch view (unified search across all KG collections)

    All methods are idempotent - safe to call multiple times.
    """

    SEMANTIC_DIM = 768 
    STRUCTURAL_DIM = 128  
    GRAPH_NAME = "video_knowledge_graph"
    VIEW_NAME = "video_kg_search_view"

    # Vector index re-indexing thresholds
    REINDEX_THRESHOLD_RATIO = 2.0  # Re-index if nLists ratio exceeds this
    MIN_DOCS_FOR_REINDEX = 100     # Minimum docs before considering re-index

    def __init__(self, db: Any):
        """Initialize the index manager.

        Args:
            db: ArangoDB database instance (standard or async)
        """
        self.db = db
        self._semantic_dim = self.SEMANTIC_DIM
        self._structural_dim = self.STRUCTURAL_DIM

    def _is_async(self) -> bool:
        """Check if the database client is async."""
        import asyncio
        return asyncio.iscoroutinefunction(self.db.collection)

    async def _ensure_analyzers(self) -> None:
        """Create text analyzers for full-text search.

        Creates:
        - text_en: Standard English text analyzer for BM25
        - wildcard_en: N-gram analyzer for partial/fuzzy matching
        """
        analyzers = [
            {
                "name": "text_en",
                "type": "text",
                "properties": {
                    "locale": "en",
                    "accent": False,
                    "case": "lower",
                    "stemming": True,
                    "stopwords": [],
                },
            },
            {
                "name": "wildcard_en",
                "type": "wildcard",
                "properties": {
                    "ngramSize": 3,
                    "analyzer": {
                        "type": "text",
                        "properties": {"locale": "en"},
                    },
                },
            },
        ]

        for analyzer in analyzers:
            try:
                if self._is_async():
                    await self.db.create_analyzer(
                        name=analyzer["name"],
                        analyzer_type=analyzer["type"],
                        properties=analyzer["properties"],
                    )
                else:
                    self.db.create_analyzer(analyzer)
                logger.info(f"[analyzer] Created: {analyzer['name']}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.debug(f"[analyzer] Already exists: {analyzer['name']}")
                else:
                    logger.warning(f"[analyzer] Failed to create {analyzer['name']}: {e}")

    async def _index_exists(self, collection_name: str, index_name: str) -> bool:
        """Check if an index already exists.

        Args:
            collection_name: Name of the collection
            index_name: Name of the index to check

        Returns:
            True if index exists, False otherwise
        """
        try:
            if self._is_async():
                coll = self.db.collection(collection_name)
                indexes = await coll.indexes()
            else:
                coll = self.db.collection(collection_name)
                indexes = coll.indexes()

            existing_names = {idx.get("name") for idx in indexes if idx.get("name")}
            return index_name in existing_names
        except Exception as e:
            logger.warning(f"Failed to check index existence: {e}")
            return False

    async def _get_index_info(self, collection_name: str, index_name: str) -> dict[str, Any] | None:
        """Get detailed metadata about a specific index.

        Args:
            collection_name: Name of the collection
            index_name: Name of the index

        Returns:
            Index metadata dict or None if not found. Includes:
            - name: Index name
            - type: Index type (vector, inverted, etc.)
            - params: Index parameters (nLists, dimension, metric for vector)
            - fields: Indexed fields
            - id: Index ID
        """
        try:
            if self._is_async():
                coll = self.db.collection(collection_name)
                indexes = await coll.indexes()
            else:
                coll = self.db.collection(collection_name)
                indexes = coll.indexes()

            for idx in indexes:
                if idx.get("name") == index_name:
                    return idx
            return None
        except Exception as e:
            logger.warning(f"Failed to get index info for {index_name}: {e}")
            return None

    async def _should_reindex_vector(
        self,
        collection_name: str,
        index_name: str,
        optimal_nlists: int,
    ) -> tuple[bool, str]:
        """Determine if a vector index should be re-indexed.

        Re-indexing is recommended when:
        1. The index doesn't exist (needs creation)
        2. The optimal nLists differs significantly from current
        3. The data volume has changed substantially

        Args:
            collection_name: Name of the collection
            index_name: Name of the vector index
            optimal_nlists: Optimal nLists based on current doc count

        Returns:
            Tuple of (should_reindex, reason)
        """
        # Get existing index info
        index_info = await self._get_index_info(collection_name, index_name)

        if index_info is None:
            return True, "Index does not exist"

        # Check if it's a vector index
        if index_info.get("type") != "vector":
            return False, "Not a vector index"

        # Get current nLists from params
        params = index_info.get("params", {})
        current_nlists = params.get("nLists", 1)

        # Don't re-index for very small collections
        n_docs = await self._get_collection_count(collection_name)
        if n_docs < self.MIN_DOCS_FOR_REINDEX:
            return False, f"Collection too small ({n_docs} docs)"

        # Check if nLists differs significantly
        if current_nlists <= 0:
            return True, "Invalid nLists value"

        ratio = max(current_nlists, optimal_nlists) / max(min(current_nlists, optimal_nlists), 1)

        if ratio >= self.REINDEX_THRESHOLD_RATIO:
            return True, (
                f"nLists changed significantly: "
                f"{current_nlists} -> {optimal_nlists} (ratio={ratio:.2f})"
            )

        return False, f"nLists acceptable ({current_nlists} vs {optimal_nlists})"

    async def _drop_index(self, collection_name: str, index_name: str) -> bool:
        """Drop an index from a collection.

        Args:
            collection_name: Name of the collection
            index_name: Name of the index to drop

        Returns:
            True if index was dropped, False otherwise
        """
        try:
            index_info = await self._get_index_info(collection_name, index_name)
            if index_info is None:
                return False

            index_id = index_info.get("id")
            if not index_id:
                return False

            if self._is_async():
                coll = self.db.collection(collection_name)
                await coll.delete_index(index_id)
            else:
                coll = self.db.collection(collection_name)
                coll.delete_index(index_id)

            logger.info(f"[DROP] Dropped index {index_name} (id={index_id}) from {collection_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to drop index {index_name}: {e}")
            return False

    async def _create_or_reindex_vector(
        self,
        collection_name: str,
        index_config: dict[str, Any],
    ) -> bool:
        """Create a vector index, re-indexing if parameters changed significantly.

        This method:
        1. Checks if index exists
        2. If exists, compares nLists with optimal value
        3. Drops and re-creates if re-indexing is needed
        4. Creates new index if doesn't exist

        Args:
            collection_name: Name of the collection
            index_config: Index configuration dict (must include 'name' and 'params')

        Returns:
            True if index was created or re-created, False on error
        """
        index_name = index_config.get("name", "unnamed")
        params = index_config.get("params", {})
        optimal_nlists = params.get("nLists", 1)

        # Check if re-indexing is needed
        should_reindex, reason = await self._should_reindex_vector(
            collection_name,
            index_name,
            optimal_nlists,
        )

        if not should_reindex:
            logger.debug(f"[SKIP] {index_name}: {reason}")
            return True

        # If index exists and needs re-indexing, drop it first
        if await self._index_exists(collection_name, index_name):
            logger.info(f"[REINDEX] {index_name}: {reason}")
            if not await self._drop_index(collection_name, index_name):
                logger.warning(f"[FAIL] Could not drop {index_name} for re-indexing")
                return False

        # Create the index
        return await self._create_index_safe(collection_name, index_config)

    def _safe_nlists(self, doc_count: int) -> int:
        """Compute safe nLists for IVF vector index.

        Ensures nLists never exceeds the number of training points.

        Args:
            doc_count: Number of documents in collection

        Returns:
            Safe nLists value
        """
        if doc_count < 2:
            return 1
        nlists = int(math.sqrt(doc_count))
        return max(1, min(nlists, doc_count))

    async def _create_index_safe(
        self,
        collection_name: str,
        index_config: dict[str, Any],
    ) -> bool:
        """Create an index if it doesn't exist.

        Args:
            collection_name: Name of the collection
            index_config: Index configuration dict

        Returns:
            True if index was created or already exists, False on error
        """
        index_name = index_config.get("name", "unnamed")

        if await self._index_exists(collection_name, index_name):
            logger.debug(f"[SKIP] Index {index_name} already exists on {collection_name}")
            return True

        try:
            coll = self.db.collection(collection_name)
            if self._is_async():
                await coll.add_index(index_config)
            else:
                coll.add_index(index_config)
            logger.info(f"[OK] Created index {index_name} on {collection_name}")
            return True
        except Exception as e:
            logger.warning(f"[WARN] Failed to create index {index_name}: {e}")
            return False

    async def _get_collection_count(self, collection_name: str) -> int:
        """Get document count for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Number of documents in collection
        """
        try:
            if self._is_async():
                coll = self.db.collection(collection_name)
                return await coll.count()
            else:
                return self.db.collection(collection_name).count()
        except Exception:
            return 0

    async def create_vector_indexes_semantic(self) -> None:
        """Create semantic vector indexes for all KG collections.

        Creates IVF vector indexes on semantic_embedding field (384-dim).
        These indexes enable fast cosine similarity search.
        """
        logger.info("[Vector Indexes — Semantic]")

        semantic_targets = [
            ("entities", "entity_semantic_idx", ["entity_type", "global_entity_id"]),
            ("events", "event_semantic_idx", ["segment_index", "start_time"]),
            ("micro_events", "micro_event_semantic_idx", ["segment_index",
                                                          "micro_index"]),
            ("communities", "community_semantic_idx", ["title", "size"]),
        ]

        for coll_name, idx_name, stored_fields in semantic_targets:
            n_docs = await self._get_collection_count(coll_name)
            nlists = self._safe_nlists(n_docs)
            n_probe = min(10, nlists)

            index_config = {
                "type": "vector",
                "name": idx_name,
                "fields": ["semantic_embedding"],
                "params": {
                    "dimension": self._semantic_dim,
                    "metric": "cosine",
                    "nLists": nlists,
                    "defaultNProbe": n_probe,
                    "trainingIterations": 40,
                },
                "storedValues": stored_fields,
            }

            await self._create_index_safe(coll_name, index_config)

    async def create_vector_indexes_structural(
        self,
        structural_dim: int | None = None,
    ) -> None:
        """Create structural (Node2Vec) vector indexes.

        Creates IVF vector indexes on structural embedding fields.
        Supports multiple structural embedding variants:
        - entity_only: Entity-only graph structure
        - entity_event: Entity-event bipartite graph
        - full: Full heterogeneous graph

        Args:
            structural_dim: Dimension of structural embeddings (default: 128)
        """
        if structural_dim:
            self._structural_dim = structural_dim

        logger.info("[Vector Indexes — Structural]")

        structural_targets = [
            (
                "entities",
                [
                    "structural_embedding_entity_only",
                    "structural_embedding_entity_event",
                    "structural_embedding_full",
                ],
            ),
            (
                "events",
                [
                    "structural_embedding_entity_event",
                    "structural_embedding_full",
                ],
            ),
            (
                "micro_events",
                [
                    "structural_embedding_entity_event",
                    "structural_embedding_full",
                ],
            ),
            (
                "communities",
                [
                    "structural_embedding_full",
                ],
            ),
        ]

        for coll_name, fields in structural_targets:
            n_docs = await self._get_collection_count(coll_name)
            nlists = self._safe_nlists(n_docs)

            for field in fields:
                idx_name = f"{coll_name}_{field}_idx"
                index_config = {
                    "type": "vector",
                    "name": idx_name,
                    "fields": [field],
                    "params": {
                        "dimension": self._structural_dim,
                        "metric": "cosine",
                        "nLists": nlists,
                        "defaultNProbe": min(10, nlists),
                        "trainingIterations": 40,
                    },
                }
                await self._create_index_safe(coll_name, index_config)

    async def create_inverted_indexes(self) -> None:
        """Create inverted indexes for BM25/TF-IDF full-text search.

        Creates inverted indexes optimized for keyword search with:
        - text_en analyzer for stemming and normalization
        - storedValues for quick field retrieval
        - optimizeTopK for efficient top-K queries
        """
        logger.info("[Inverted Indexes]")

        inverted_configs = [
            {
                "collection": "entities",
                "name": "entities_inverted_idx",
                "fields": [
                    {"name": "video_id"},
                    {"name": "entity_type"},
                    {"name": "global_entity_id"},
                    {"name": "entity_name", "analyzer": "text_en"},
                ],
                "storedValues": ["desc", "first_seen_segment", "last_seen_segment"],
                "optimizeTopK": ["BM25(@doc) DESC", "TFIDF(@doc) DESC"],
            },
            {
                "collection": "events",
                "name": "events_inverted_idx",
                "fields": [
                    {"name": "video_id"},
                    {"name": "segment_index"},
                    {"name": "caption", "analyzer": "text_en"},
                ],
                "storedValues": ["start_time", "end_time", "start_sec", "end_sec"],
                "optimizeTopK": ["BM25(@doc) DESC", "TFIDF(@doc) DESC"],
            },
            {
                "collection": "micro_events",
                "name": "micro_events_inverted_idx",
                "fields": [
                    {"name": "video_id"},
                    {"name": "segment_index"},
                    {"name": "parent_event_key"},
                    {"name": "text", "analyzer": "text_en"},
                    {"name": "related_caption_context", "analyzer": "text_en"},
                ],
                "storedValues": [
                    "start_time",
                    "end_time",
                    "start_secs",
                    "end_secs",
                    "micro_index",
                ],
                "optimizeTopK": ["BM25(@doc) DESC", "TFIDF(@doc) DESC"],
            },
            {
                "collection": "communities",
                "name": "communities_inverted_idx",
                "fields": [
                    {"name": "video_id"},
                    {"name": "title", "analyzer": "text_en"},
                    {"name": "summary", "analyzer": "text_en"},
                ],
                "storedValues": ["size", "comm_idx"],
                "optimizeTopK": ["BM25(@doc) DESC"],
            },
        ]

        for cfg in inverted_configs:
            index_config: dict[str, Any] = {
                "type": "inverted",
                "name": cfg["name"],
                "fields": cfg["fields"],
                "storedValues": cfg["storedValues"],
            }
            if cfg.get("optimizeTopK"):
                index_config["optimizeTopK"] = cfg["optimizeTopK"]

            await self._create_index_safe(cfg["collection"], index_config)

    async def create_mdi_indexes(self) -> None:
        """Create MDI (Multi-Dimensional Interval) indexes.

        MDI indexes enable efficient time-range queries on events.
        Useful for queries like "find events between 10s and 30s".
        """
        logger.info("[MDI Indexes]")

        mdi_configs = [
            {
                "collection": "events",
                "name": "events_time_mdi_idx",
                "fields": ["start_sec", "end_sec"],
                "fieldValueTypes": "double",
            },
            {
                "collection": "micro_events",
                "name": "micro_events_time_mdi_idx",
                "fields": ["start_secs", "end_secs"],
                "fieldValueTypes": "double",
            },
        ]

        for cfg in mdi_configs:
            index_config = {
                "type": "mdi",
                "name": cfg["name"],
                "fields": cfg["fields"],
                "fieldValueTypes": cfg["fieldValueTypes"],
            }
            await self._create_index_safe(cfg["collection"], index_config)

    async def create_persistent_indexes(self) -> None:
        """Create persistent indexes for fast lookups.

        These indexes speed up equality and range queries.
        """
        logger.info("[Persistent Indexes]")

        persistent_configs = [
            {
                "collection": "entities",
                "name": "entities_video_id_idx",
                "fields": ["video_id"],
                "sparse": True,
            },
            {
                "collection": "events",
                "name": "events_video_segment_idx",
                "fields": ["video_id", "segment_index"],
                "sparse": True,
            },
            {
                "collection": "micro_events",
                "name": "micro_events_video_segment_idx",
                "fields": ["video_id", "segment_index", "micro_index"],
                "sparse": True,
            },
            {
                "collection": "communities",
                "name": "communities_video_id_idx",
                "fields": ["video_id"],
                "sparse": True,
            },
        ]

        for cfg in persistent_configs:
            index_config = {
                "type": "persistent",
                "name": cfg["name"],
                "fields": cfg["fields"],
                "sparse": cfg.get("sparse", False),
            }
            await self._create_index_safe(cfg["collection"], index_config)

    async def create_arangosearch_view(self) -> None:
        """Create ArangoSearch view for unified full-text search.

        Creates a view that links all KG collections for cross-collection
        full-text search using BM25 scoring.
        """
        logger.info(f"[ArangoSearch View] {self.VIEW_NAME}")

        # Check if view already exists
        try:
            if self._is_async():
                views = await self.db.views()
            else:
                views = self.db.views()

            existing_views = {v["name"] for v in views}
            if self.VIEW_NAME in existing_views:
                logger.debug(f"View {self.VIEW_NAME} already exists")
                return
        except Exception as e:
            logger.warning(f"Failed to check existing views: {e}")

        view_links = {
            "entities": {
                "analyzers": ["text_en"],
                "fields": {
                    "entity_name": {"analyzers": ["text_en"]},
                    "entity_type": {"analyzers": ["text_en"]},
                    "desc": {"analyzers": ["text_en"]},
                },
                "includeAllFields": False,
                "storeValues": "none",
                "trackListPositions": False,
            },
            "events": {
                "analyzers": ["text_en"],
                "fields": {
                    "caption": {"analyzers": ["text_en"]},
                },
                "includeAllFields": False,
            },
            "micro_events": {
                "analyzers": ["text_en"],
                "fields": {
                    "text": {"analyzers": ["text_en"]},
                    "related_caption_context": {"analyzers": ["text_en"]},
                },
                "includeAllFields": False,
            },
            "communities": {
                "analyzers": ["text_en"],
                "fields": {
                    "title": {"analyzers": ["text_en"]},
                    "summary": {"analyzers": ["text_en"]},
                },
                "includeAllFields": False,
            },
        }

        try:
            if self._is_async():
                await self.db.create_arangosearch_view(
                    name=self.VIEW_NAME,
                    properties={
                        "links": view_links,
                        "primarySort": [],
                        "storedValues": [],
                        "consolidationIntervalMsec": 1000,
                    },
                )
            else:
                self.db.create_arangosearch_view(
                    name=self.VIEW_NAME,
                    properties={
                        "links": view_links,
                        "primarySort": [],
                        "storedValues": [],
                        "consolidationIntervalMsec": 1000,
                    },
                )
            logger.info(f"Created ArangoSearch view {self.VIEW_NAME}")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.debug(f"View {self.VIEW_NAME} already exists")
            else:
                logger.warning(f"Failed to create view: {e}")

    async def ensure_all_indexes(self, structural_dim: int | None = None) -> dict[str, bool]:
        """Create all required indexes for KG collections.

        This is the main entry point for index creation during lifespan.
        All operations are idempotent.

        Args:
            structural_dim: Dimension of structural embeddings (optional)

        Returns:
            Dict mapping index type to success status
        """
        logger.info("=== Ensuring all ArangoDB indexes ===")

        results = {}

        try:
            await self._ensure_analyzers()
            results["analyzers"] = True
        except Exception as e:
            logger.error(f"Failed to create analyzers: {e}")
            results["analyzers"] = False

        try:
            await self.create_vector_indexes_semantic()
            results["vector_semantic"] = True
        except Exception as e:
            logger.error(f"Failed to create semantic vector indexes: {e}")
            results["vector_semantic"] = False

        try:
            await self.create_vector_indexes_structural(structural_dim)
            results["vector_structural"] = True
        except Exception as e:
            logger.error(f"Failed to create structural vector indexes: {e}")
            results["vector_structural"] = False

        try:
            await self.create_inverted_indexes()
            results["inverted"] = True
        except Exception as e:
            logger.error(f"Failed to create inverted indexes: {e}")
            results["inverted"] = False

        try:
            await self.create_mdi_indexes()
            results["mdi"] = True
        except Exception as e:
            logger.error(f"Failed to create MDI indexes: {e}")
            results["mdi"] = False

        try:
            await self.create_persistent_indexes()
            results["persistent"] = True
        except Exception as e:
            logger.error(f"Failed to create persistent indexes: {e}")
            results["persistent"] = False

        try:
            await self.create_arangosearch_view()
            results["arangosearch_view"] = True
        except Exception as e:
            logger.error(f"Failed to create ArangoSearch view: {e}")
            results["arangosearch_view"] = False

        logger.info(f"=== Index creation complete: {results} ===")
        return results

    async def detect_structural_dim(self) -> int:
        """Detect structural embedding dimension from existing data.

        Returns:
            Detected dimension or default (128)
        """
        try:
            aql = """
            FOR e IN entities
                FILTER e.structural_embedding_full != null
                LIMIT 1
                RETURN LENGTH(e.structural_embedding_full)
            """
            if self._is_async():
                cursor = await self.db.aql.execute(aql)
                async for dim in cursor:
                    return dim
            else:
                cursor = self.db.aql.execute(aql)
                for dim in cursor:
                    return dim
        except Exception:
            pass
        return self.STRUCTURAL_DIM
    
    async def get_index_stats(self) -> dict[str, Any]:
        """Get statistics about existing indexes.

        Returns:
            Dict with index statistics per collection
        """
        stats = {}
        collections = ["entities", "events", "micro_events", "communities"]

        for coll_name in collections:
            try:
                if self._is_async():
                    coll = self.db.collection(coll_name)
                    indexes = await coll.indexes()
                else:
                    coll = self.db.collection(coll_name)
                    indexes = coll.indexes()

                stats[coll_name] = {
                    "count": len(indexes),
                    "names": [idx.get("name") for idx in indexes if idx.get("name")],
                    "types": [idx.get("type") for idx in indexes],
                }
            except Exception as e:
                stats[coll_name] = {"error": str(e)}

        return stats
