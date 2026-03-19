"""ArangoDB storage clients for Knowledge Graph operations."""

from .client import ArangoKGClient
from .index_manager import ArangoIndexManager
from .schema import (
    KGCommunityResult,
    KGEntityResult,
    KGEventResult,
    KGMicroEventResult,
    KGRagResult,
    KGTraversalResult,
)

__all__ = [
    "ArangoKGClient",
    "ArangoIndexManager",
    "KGEntityResult",
    "KGEventResult",
    "KGMicroEventResult",
    "KGCommunityResult",
    "KGRagResult",
    "KGTraversalResult",
]
