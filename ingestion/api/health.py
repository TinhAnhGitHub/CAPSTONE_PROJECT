from fastapi import APIRouter, Depends, status
from enum import Enum
from typing import Any, Optional
from pydantic import Field, BaseModel
from datetime import datetime
from loguru import logger
from sqlalchemy import text
from core.dependencies.application import get_artifact_tracker, get_storage_client

from core.clients.base import BaseMilvusClient
from core.pipeline.tracker import ArtifactTracker
from core.storage import StorageClient
from core.clients.base import ClientConfig
from core.config.milvus_index_config import MilvusIndexBaseConfig
from core.clients.asr_client import ASRClient
from core.clients.autoshot_client import AutoshotClient
from core.clients.llm_client import LLMClient
from core.clients.image_embed_client import ImageEmbeddingClient
from core.clients.text_embed_client import TextEmbeddingClient
from core.pipeline.service_registry import ConsulServiceRegistry
from core.clients.milvus_client import (
    ImageMilvusClient,
    SegmentCaptionEmbeddingMilvusClient
)
from core.app_state import AppState

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
class ComponentHealth(BaseModel):
    """Health status for a single component."""
    name: str
    status: HealthStatus
    response_time_ms: Optional[float] = None
    details: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    checked_at: datetime = Field(default_factory=datetime.now)


class OverallHealth(BaseModel):
    """Overall system health status."""
    status: HealthStatus
    timestamp: datetime = Field(default_factory=datetime.now)
    components: dict[str, ComponentHealth]
    summary: dict[str, Any]


router = APIRouter(
    prefix='/pipeline_check', tags=['check']
)

async def check_component(
    name: str,
    check_func,
    *args,
    **kwargs
) -> ComponentHealth:
    """Generic component health check wrapper."""
    start_time = datetime.now()
    
    result = await check_func(*args, **kwargs)
    response_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return ComponentHealth(
        name=name,
        status=HealthStatus.HEALTHY,
        response_time_ms=round(response_time, 2),
        details=result if isinstance(result, dict) else {"result": str(result)}
    )
    

async def check_database(tracker: ArtifactTracker) -> dict[str, Any]:
    async with tracker.get_session() as session:
        result = await session.execute(text("SELECT 1")) #type:ignore
        _ = result.scalar()
        
        
        tables_query = text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        
        result = await session.execute(tables_query)
        table_names = [row[0] for row in result.fetchall()]
        
        return {
            "connected": True,
            "database_url": tracker.database_url.split('@')[-1],  
            "tables": len(table_names),
            "table_names": table_names
        }
    

async def check_storage(storage: StorageClient) -> dict[str, Any]:
    """Check MinIO/S3 storage connectivity."""
    try:
        buckets = [b.name for b in storage.client.list_buckets()]
        
        return {
            "connected": True,
            "endpoint": f"{storage.settings.host}:{storage.settings.port}",
            "buckets": buckets,
            "bucket_count": len(buckets)
        }
    except Exception as e:
        raise RuntimeError(f"Storage connection failed: {e}")
    


async def check_consul_connection(config: ClientConfig) -> dict[str, Any]:
    print(f"{config.model_dump(mode='json')}")
    consul = ConsulServiceRegistry(
        host=config.consul_host,
        port=config.consul_port
    )
    
    
    services = await consul.consul.catalog.services()
    service_names = list(services[1].keys()) if services else []
    
    return {
        "connected": True,
        "consul_url": f"{config.consul_host}:{config.consul_port}",
        "services_registered": len(service_names),
        "service_names": service_names
    }




async def check_microservice(
    client_class,
    service_name: str,
    config: ClientConfig
) -> dict[str, Any]:
    """Check microservice connectivity and status."""
    async with client_class(config=config) as client:
        service_url = await client.get_service_url()
        print(f"{service_url=}")

        
        try:
            service_status = await client.get_status()
        except:
            service_status = {"status": "unknown"}
        
        
        models = await client.list_models()
        
        print(f"{service_status=}")
        print(f"{models=}")
        return {
            "service_url": service_url,
            "service_name": service_name,
            "status": service_status,
            "available_models": models
        }
    
async def check_milvus(
    client_instance,
) -> dict[str, Any]:
    await client_instance.connect()
    has_collection = await client_instance._client.has_collection(
        client_instance.collection_name
    )

    stats = {}
    if has_collection:
        stats = await client_instance.get_collection_stats()
    await client_instance.close()

    return {
        "connected": True,
        "database": client_instance.db_name,
        "collection_name": client_instance.collection_name,
        "collection_exists": has_collection,
        "stats": stats
    }


@router.get(
    '/',
    status_code=status.HTTP_200_OK,
    summary="Overall system health check",
    description="Comprehensive health check of all system components"
)
async def health_check(
    tracker: ArtifactTracker = Depends(get_artifact_tracker),
    storage: StorageClient = Depends(get_storage_client),
):  
    asr_config = AppState().base_client_config
    components = {}
    components["database"] = await check_component(
        "database",
        check_database,
        tracker
    )
    
    # Storage
    components["storage"] = await check_component(
        "storage",
        check_storage,
        storage
    )
    
    # Consul
    components["consul"] = await check_component(
        "consul",
        check_consul_connection,
        asr_config
    )
    
    # Determine overall status
    unhealthy = [c for c in components.values() if c.status == HealthStatus.UNHEALTHY]
    
    if unhealthy:
        overall_status = HealthStatus.UNHEALTHY
    else:
        overall_status = HealthStatus.HEALTHY
    
    summary = {
        "total_components": len(components),
        "healthy": len([c for c in components.values() if c.status == HealthStatus.HEALTHY]),
        "unhealthy": len(unhealthy)
    }
    
    return OverallHealth(
        status=overall_status,
        components=components,
        summary=summary
    )

@router.get(
    "/database",
    response_model=ComponentHealth,
    summary="Database health check"
)
async def check_database_health(
    tracker: ArtifactTracker = Depends(get_artifact_tracker)
) -> ComponentHealth:
    return await check_component("database", check_database, tracker)


@router.get(
    "/storage",
    response_model=ComponentHealth,
    summary="Storage health check"
)
async def check_storage_health(
    storage: StorageClient = Depends(get_storage_client)
) -> ComponentHealth:
    return await check_component("storage", check_storage, storage)


@router.get(
    "/consul",
    response_model=ComponentHealth,
    summary="Consul service discovery health check"
)
async def check_consul_health() -> ComponentHealth:
    config = AppState().base_client_config
    return await check_component("consul", check_consul_connection, config)


@router.get(
    "/services/asr",
    response_model=ComponentHealth,
    summary="ASR service health check"
)
async def check_asr_service() -> ComponentHealth:
    config = AppState().base_client_config
    return await check_component(
        "asr-service",
        check_microservice,
        ASRClient,
        "service-asr",
        config
    )


@router.get(
    "/services/autoshot",
    response_model=ComponentHealth,
    summary="Autoshot service health check"
)
async def check_autoshot_service() -> ComponentHealth:
    config = AppState().base_client_config
    """Check Autoshot microservice connectivity and status."""
    return await check_component(
        "service-autoshot",
        check_microservice,
        AutoshotClient,
        "service-autoshot",
        config
    )



@router.get(
    "/services/llm",
    response_model=ComponentHealth,
    summary="LLM service health check"
)
async def check_llm_service() -> ComponentHealth:
    config = AppState().base_client_config
    return await check_component(
        "llm-service",
        check_microservice,
        LLMClient,
        "llm-service",
        config
    )


@router.get(
    "/services/image-embedding",
    response_model=ComponentHealth,
    summary="Image embedding service health check"
)
async def check_image_embedding_service() -> ComponentHealth:
    
    config = AppState().base_client_config
    return await check_component(
        "image-embedding-service",
        check_microservice,
        ImageEmbeddingClient,
        "image_embedding",
        config
    )


@router.get(
    "/services/text-embedding",
    response_model=ComponentHealth,
    summary="Text embedding service health check"
)
async def check_text_embedding_service() -> ComponentHealth:
    """Check text embedding microservice connectivity and status."""
    config = AppState().base_client_config
    return await check_component(
        "text-embedding-service",
        check_microservice,
        TextEmbeddingClient,
        "text_embedding",
        config
    )


@router.get(
    "/milvus/image-embeddings",
    response_model=ComponentHealth,
    summary="Image embeddings Milvus health check"
)
async def check_image_embeddings_milvus() -> ComponentHealth:
    """Check image embeddings Milvus collection status."""

    milvus_config_collection = AppState().image_milvus_client
    
    return await check_component(
        "milvus-image-embeddings",
        check_milvus,
        milvus_config_collection
    )


@router.get(
    "/milvus/segment-caption-embeddings",
    response_model=ComponentHealth,
    summary="Segment caption embeddings Milvus health check"
)
async def check_segment_caption_milvus() -> ComponentHealth:
    """Check segment caption embeddings Milvus collection status."""

    milvus_config_collection = AppState().seg_milvus_client    
    return await check_component(
        "milvus-segment-caption-embeddings",
        check_milvus,
        milvus_config_collection
    )
