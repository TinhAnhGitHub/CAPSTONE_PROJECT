"""ArangoDB client for knowledge graph operations.

This module provides database setup and connection utilities for ArangoDB.
The setup creates the database, collections, and named graph structure.

Note: Vector indexes are NOT created here - they will be created in
videodeepsearch during the lifespan setup after data is loaded.
"""

from __future__ import annotations

from typing import Any

from arango.client import ArangoClient
from arango.database import StandardDatabase
from loguru import logger

from .config import ArangoConfig
from .exception import ArangoConnectionError


VERTEX_COLLECTIONS = [
    "videos",          # One doc per ingested video
    "entities",        # CanonicalEntity
    "events",          # EventNode (segment-level)
    "micro_events",    # MicroEventNode
    "communities",     # CommunityDoc
]

EDGE_COLLECTIONS = [
    "entity_relations",        # entity <-> entity
    "event_sequences",         # event  <-> event
    "event_entities",          # event  <-> entity
    "micro_event_sequences",   # micro_event <-> micro_event
    "micro_event_parents",     # micro_event -> event (parent)
    "micro_event_entities",    # micro_event <-> entity
    "community_members",       # entity -> community
    "event_communities",       # event  -> community
]


class ArangoStorageClient:
    """ArangoDB client for knowledge graph operations.

    Provides connection management, database setup, and basic CRUD operations.
    """

    def __init__(self, config: ArangoConfig):
        self._config = config
        self._client: ArangoClient | None = None
        self._db: StandardDatabase | None = None

    def connect(self) -> StandardDatabase:
        """Connect to ArangoDB and return the configured database.

        Returns:
            StandardDatabase instance for the configured database
        """
        self._client = ArangoClient(hosts=self._config.host)

        sys_db = self._client.db(
            "_system",
            username=self._config.username,
            password=self._config.password,
        )

        if not sys_db.has_database(self._config.database):
            sys_db.create_database(self._config.database)
            logger.info(f"Created ArangoDB database: {self._config.database}")

        self._db = self._client.db(
            self._config.database,
            username=self._config.username,
            password=self._config.password,
        )

        logger.info(f"Connected to ArangoDB database: {self._config.database}")
        return self._db

    def disconnect(self) -> None:
        """Close the ArangoDB connection."""
        self._db = None
        self._client = None
        logger.info("Disconnected from ArangoDB")

    def get_db(self) -> StandardDatabase:
        """Get the database connection, connecting if necessary.

        Returns:
            StandardDatabase instance for the configured database
        """
        if self._db is None:
            self.connect()
        return self._db  # type: ignore

    def setup_database(self) -> StandardDatabase:
        """Set up the complete database structure.

        Creates:
        - Database (if not exists)
        - Vertex collections
        - Edge collections
        - Named graph with edge definitions

        Returns:
            StandardDatabase instance for the configured database
        """
        db = self.get_db()
        graph_name = self._config.graph_name

        for name in VERTEX_COLLECTIONS:
            if not db.has_collection(name):
                db.create_collection(name)
                logger.info(f"Created vertex collection: {name}")

        for name in EDGE_COLLECTIONS:
            if not db.has_collection(name):
                db.create_collection(name, edge=True)
                logger.info(f"Created edge collection: {name}")

        if not db.has_graph(graph_name):
            graph = db.create_graph(graph_name)

            graph.create_edge_definition( #type:ignore
                edge_collection="entity_relations",
                from_vertex_collections=["entities"],
                to_vertex_collections=["entities"],
            )

            graph.create_edge_definition( #type:ignore
                edge_collection="event_sequences",
                from_vertex_collections=["events"],
                to_vertex_collections=["events"],
            )

            graph.create_edge_definition( #type:ignore
                edge_collection="event_entities",
                from_vertex_collections=["events"],
                to_vertex_collections=["entities"],
            )

            graph.create_edge_definition( #type:ignore
                edge_collection="micro_event_sequences",
                from_vertex_collections=["micro_events"],
                to_vertex_collections=["micro_events"],
            )

            graph.create_edge_definition( #type:ignore
                edge_collection="micro_event_parents",
                from_vertex_collections=["micro_events"],
                to_vertex_collections=["events"],
            )

            graph.create_edge_definition( #type:ignore
                edge_collection="micro_event_entities",
                from_vertex_collections=["micro_events"],
                to_vertex_collections=["entities"],
            )
 
            graph.create_edge_definition( #type:ignore
                edge_collection="community_members",
                from_vertex_collections=["entities"],
                to_vertex_collections=["communities"],
            )

            graph.create_edge_definition( #type:ignore
                edge_collection="event_communities",
                from_vertex_collections=["events"],
                to_vertex_collections=["communities"],
            )

            logger.info(f"Created named graph: {graph_name}")

        logger.info("ArangoDB setup complete (no vector indexes)")
        return db

    def insert_document(
        self,
        collection: str,
        document: dict[str, Any],
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Insert a document into a collection.

        Args:
            collection: Collection name
            document: Document to insert
            overwrite: Whether to overwrite if document exists

        Returns:
            Inserted document metadata
        """
        db = self.get_db()
        coll = db.collection(collection)

        if overwrite:
            return coll.insert(document, overwrite_mode="replace") #type:ignore
        return coll.insert(document) #type:ignore

    def insert_documents(
        self,
        collection: str,
        documents: list[dict[str, Any]],
        overwrite: bool = False,
    ) -> list[dict[str, Any]]:
        """Insert multiple documents into a collection.

        Args:
            collection: Collection name
            documents: Documents to insert
            overwrite: Whether to overwrite if documents exist

        Returns:
            List of inserted document metadata
        """
        if not documents:
            return []

        db = self.get_db()
        coll = db.collection(collection)

        if overwrite:
            return coll.insert_many(documents, overwrite_mode="replace") #type:ignore
        return coll.insert_many(documents) #type:ignore

    def query(
        self, query: str, bind_vars: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute an AQL query.

        Args:
            query: AQL query string
            bind_vars: Bind variables

        Returns:
            Query results
        """
        db = self.get_db()
        cursor = db.aql.execute(query, bind_vars=bind_vars or {})
        return list(cursor) #type:ignore

    def truncate_collection(self, collection: str) -> None:
        """Truncate a collection.

        Args:
            collection: Collection name
        """
        db = self.get_db()
        if db.has_collection(collection):
            db.collection(collection).truncate()
            logger.info(f"Truncated collection: {collection}")

    def __enter__(self) -> "ArangoStorageClient":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()