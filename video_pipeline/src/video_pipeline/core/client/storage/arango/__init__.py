"""ArangoDB storage client for knowledge graph operations."""

from .client import ArangoStorageClient, VERTEX_COLLECTIONS, EDGE_COLLECTIONS
from .config import ArangoConfig, ArangoIndexConfig, ArangoInvertedIndexConfig

__all__ = [
    "ArangoStorageClient",
    "ArangoConfig",
    "ArangoIndexConfig",
    "ArangoInvertedIndexConfig",
    "VERTEX_COLLECTIONS",
    "EDGE_COLLECTIONS",
]