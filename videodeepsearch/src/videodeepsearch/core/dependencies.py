from __future__ import annotations
from typing import TYPE_CHECKING, Annotated
from fastapi import Depends, Request

if TYPE_CHECKING:
    from arango.database import StandardDatabase
    from videodeepsearch.clients.storage.postgre import PostgresClient
    from videodeepsearch.clients.storage.minio import MinioStorageClient
    from videodeepsearch.clients.storage.qdrant import ImageQdrantClient, SegmentQdrantClient, AudioQdrantClient
    from videodeepsearch.clients.storage.elasticsearch import ElasticsearchOCRClient
    from videodeepsearch.clients.inference import QwenVLEmbeddingClient, MMBertClient, SpladeClient
    from videodeepsearch.agent.member.worker.spawn_toolkit import WorkerModel
    from agno.models.base import Model

def get_postgres(request: Request) -> "PostgresClient":
    return request.app.state.postgres_client


def get_minio(request: Request) -> "MinioStorageClient":
    return request.app.state.minio_client


def get_image_qdrant(request: Request) -> "ImageQdrantClient":
    return request.app.state.image_qdrant_client


def get_segment_qdrant(request: Request) -> "SegmentQdrantClient":
    return request.app.state.segment_qdrant_client


def get_audio_qdrant(request: Request) -> "AudioQdrantClient":
    return request.app.state.audio_qdrant_client


def get_es_ocr(request: Request) -> "ElasticsearchOCRClient":
    return request.app.state.es_ocr_client

def get_arango_db(request: Request) -> "StandardDatabase":
    return request.app.state.arango_db

def get_qwenvl(request: Request) -> "QwenVLEmbeddingClient":
    return request.app.state.qwenvl_client

def get_mmbert(request: Request) -> "MMBertClient":
    return request.app.state.mmbert_client

def get_splade(request: Request) -> "SpladeClient":
    return request.app.state.splade_client

def get_models(request: Request) -> dict[str, "Model"]:
    return request.app.state.models

def get_worker_models(request: Request) -> dict[str, "WorkerModel"]:
    return request.app.state.worker_models

PostgresDep = Annotated["PostgresClient", Depends(get_postgres)]
MinioDep = Annotated["MinioStorageClient", Depends(get_minio)]
ImageQdrantDep = Annotated["ImageQdrantClient", Depends(get_image_qdrant)]
SegmentQdrantDep = Annotated["SegmentQdrantClient", Depends(get_segment_qdrant)]
AudioQdrantDep = Annotated["AudioQdrantClient", Depends(get_audio_qdrant)]
EsOcrDep = Annotated["ElasticsearchOCRClient", Depends(get_es_ocr)]
ArangoDep = Annotated["StandardDatabase", Depends(get_arango_db)]

QwenVlDep = Annotated["QwenVLEmbeddingClient", Depends(get_qwenvl)]
MMBertDep = Annotated["MMBertClient", Depends(get_mmbert)]
SpladeDep = Annotated["SpladeClient", Depends(get_splade)]

ModelsDep = Annotated[dict[str, "Model"], Depends(get_models)]
WorkerModelsDep = Annotated[dict[str, "WorkerModel"], Depends(get_worker_models)]