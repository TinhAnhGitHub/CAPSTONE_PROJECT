"""ArangoDB storage client for knowledge graph operations."""

from .client import ArangoStorageClient, VERTEX_COLLECTIONS, EDGE_COLLECTIONS
from .config import ArangoConfig, ArangoIndexConfig, ArangoInvertedIndexConfig
from .exception import (
    ArangoClientError,
    ArangoConnectionError,
    ArangoQueryError,
    ArangoIndexError,
)

__all__ = [
    "ArangoStorageClient",
    "ArangoConfig",
    "ArangoIndexConfig",
    "ArangoInvertedIndexConfig",
    "ArangoClientError",
    "ArangoConnectionError",
    "ArangoQueryError",
    "ArangoIndexError",
    "VERTEX_COLLECTIONS",
    "EDGE_COLLECTIONS",
]