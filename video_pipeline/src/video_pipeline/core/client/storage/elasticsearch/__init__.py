"""Elasticsearch client module."""

from video_pipeline.core.client.storage.elasticsearch.config import ElasticsearchSettings
from video_pipeline.core.client.storage.elasticsearch.client import ElasticsearchOCRClient
from video_pipeline.core.client.storage.elasticsearch.utils import (
    clean_ocr_text,
    get_ocr_index_mapping,
)

__all__ = [
    "ElasticsearchSettings",
    "ElasticsearchOCRClient",
    "clean_ocr_text",
    "get_ocr_index_mapping",
]