"""PostgreSQL client for artifact metadata."""

from videodeepsearch.clients.storage.postgre.client import PostgresClient
from videodeepsearch.clients.storage.postgre.schema import (
    ArtifactLineageSchema,
    ArtifactMetadata,
    ArtifactSchema,
)

__all__ = [
    "PostgresClient",
    "ArtifactMetadata",
    "ArtifactSchema",
    "ArtifactLineageSchema",
]