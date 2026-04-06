from __future__ import annotations

from collections.abc import Callable
from typing import Any

from arango.database import StandardDatabase

from videodeepsearch.clients.storage.elasticsearch import ElasticsearchOCRClient
from videodeepsearch.clients.storage.minio import MinioStorageClient
from videodeepsearch.clients.storage.postgre import PostgresClient
from videodeepsearch.toolkit.kg_retrieval import KGSearchToolkit
from videodeepsearch.toolkit.llm import LLMToolkit
from videodeepsearch.toolkit.ocr import OCRSearchToolkit
from videodeepsearch.toolkit.search import VideoSearchToolkit
from videodeepsearch.toolkit.utility import UtilityToolkit
from videodeepsearch.toolkit.video_metadata import VideoMetadataToolkit


ToolkitFactory = Callable[[], Any]


def make_search_factory(
    image_qdrant_client,
    segment_qdrant_client,
    audio_qdrant_client,
    qwenvl_client,
    mmbert_client,
    splade_client,
    user_id: str | None = None,
    video_ids: list[str] | None = None,
) -> ToolkitFactory:
    def factory() -> VideoSearchToolkit:
        return VideoSearchToolkit(
            image_qdrant_client=image_qdrant_client,
            segment_qdrant_client=segment_qdrant_client,
            audio_qdrant_client=audio_qdrant_client,
            qwenvl_client=qwenvl_client,
            mmbert_client=mmbert_client,
            splade_client=splade_client,
            user_id=user_id,
            video_ids=video_ids,
        )
    return factory


def make_utility_factory(
    postgres_client: PostgresClient,
    minio_client: MinioStorageClient,
) -> ToolkitFactory:
    def factory() -> UtilityToolkit:
        return UtilityToolkit(
            postgres_client=postgres_client,
            minio_client=minio_client,
        )
    return factory


def make_video_metadata_factory(
    postgres_client: PostgresClient,
    minio_client: MinioStorageClient,
) -> ToolkitFactory:
    def factory() -> VideoMetadataToolkit:
        return VideoMetadataToolkit(
            postgres_client=postgres_client,
            minio_client=minio_client,
        )
    return factory


def make_ocr_factory(
    es_ocr_client: ElasticsearchOCRClient,
    mmbert_client,
    user_id: str | None = None,
    video_ids: list[str] | None = None,
) -> ToolkitFactory:
    def factory() -> OCRSearchToolkit:
        return OCRSearchToolkit(
            es_ocr_client=es_ocr_client,
            mmbert_client=mmbert_client,
            user_id=user_id,
            video_ids=video_ids,
        )
    return factory


def make_llm_factory(llm_client) -> ToolkitFactory:
    def factory() -> LLMToolkit:
        return LLMToolkit(llm_client=llm_client)
    return factory


def make_kg_factory(
    arango_db: StandardDatabase,
    mmbert_client,
    user_id: str | None = None,
    video_ids: list[str] | None = None,
) -> ToolkitFactory:
    def factory() -> KGSearchToolkit:
        return KGSearchToolkit(
            arango_db=arango_db,
            mmbert_client=mmbert_client,
            user_id=user_id,
            video_ids=video_ids,
        )
    return factory


__all__ = [
    "ToolkitFactory",
    "make_search_factory",
    "make_utility_factory",
    "make_video_metadata_factory",
    "make_ocr_factory",
    "make_llm_factory",
    "make_kg_factory",
]
