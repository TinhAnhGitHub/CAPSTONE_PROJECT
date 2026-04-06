"""Elasticsearch storage client module."""

from videodeepsearch.clients.storage.elasticsearch.schema import ElasticsearchConfig
from videodeepsearch.clients.storage.elasticsearch.client import (
    ElasticsearchOCRClient,
    OCRSearchResult,
)

__all__ = [
    "ElasticsearchConfig",
    "ElasticsearchOCRClient",
    "OCRSearchResult",
]