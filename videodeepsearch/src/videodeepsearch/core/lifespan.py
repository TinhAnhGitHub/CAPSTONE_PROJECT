"""FastAPI lifespan for initializing all clients and models."""

from __future__ import annotations

from contextlib import asynccontextmanager

from arango.client import ArangoClient
from fastapi import FastAPI
from loguru import logger
from sqlalchemy import text
from agno.db.postgres import AsyncPostgresDb
from agno.models.openrouter import OpenRouter, OpenRouterResponses

from videodeepsearch.agent.member.worker.spawn_toolkit import WorkerModel
from videodeepsearch.clients.inference import (
    MMBertClient,
    MMBertConfig,
    QwenVLEmbeddingClient,
    QwenVLEmbeddingConfig,
    SpladeClient,
    SpladeConfig,
)
from videodeepsearch.clients.storage.elasticsearch import ElasticsearchOCRClient
from videodeepsearch.clients.storage.minio import MinioStorageClient
from videodeepsearch.clients.storage.postgre import PostgresClient
from videodeepsearch.clients.storage.qdrant import AudioQdrantClient, ImageQdrantClient, SegmentQdrantClient
from videodeepsearch.core.settings import settings


def _init_mlflow():
    if not settings.mlflow.enabled:
        logger.info("MLflow tracing disabled")
        return

    import mlflow

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)
    mlflow.agno.autolog() #type:ignore
    logger.info(f"MLflow tracing enabled: {settings.mlflow.tracking_uri}, experiment: {settings.mlflow.experiment_name}")



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all clients and models into app.state."""
    logger.info("Initializing VideoDeepSearch...")

    _init_mlflow()

    app.state.postgres_client = PostgresClient(
        database_url=settings.storage.postgres.connection_url
    )
    async with app.state.postgres_client.get_session() as session:
        result = await session.execute(text("SELECT version();"))
        version = result.scalar_one()
        logger.info(f"PostgreSQL connected: {version}")

    # Initialize agno AsyncPostgresDb for Team session storage
    app.state.agno_db = AsyncPostgresDb(
        db_url=settings.storage.postgres.connection_url,
        create_schema=True,
    )
    logger.info("Agno AsyncPostgresDb initialized for session storage")

    app.state.minio_client = MinioStorageClient(
        host=settings.storage.minio.host,
        port=settings.storage.minio.port,
        access_key=settings.storage.minio.access_key,
        secret_key=settings.storage.minio.secret_key,
        secure=settings.storage.minio.secure,
    )
    logger.info("MinIO client initialized")

    qdrant_cfg = settings.storage.qdrant
    app.state.image_qdrant_client = ImageQdrantClient(
        host=qdrant_cfg.host,
        port=qdrant_cfg.port,
        collection_name=qdrant_cfg.collection_name,
        grpc_port=qdrant_cfg.grpc_port,
        prefer_grpc=qdrant_cfg.prefer_grpc,
    )
    app.state.segment_qdrant_client = SegmentQdrantClient(
        host=qdrant_cfg.host,
        port=qdrant_cfg.port,
        collection_name=qdrant_cfg.collection_name,
        grpc_port=qdrant_cfg.grpc_port,
        prefer_grpc=qdrant_cfg.prefer_grpc,
    )
    app.state.audio_qdrant_client = AudioQdrantClient(
        host=qdrant_cfg.host,
        port=qdrant_cfg.port,
        collection_name=qdrant_cfg.collection_name,
        grpc_port=qdrant_cfg.grpc_port,
        prefer_grpc=qdrant_cfg.prefer_grpc,
    )
    logger.info(f"Qdrant clients initialized: {qdrant_cfg.host}:{qdrant_cfg.port}")

    es_cfg = settings.storage.elasticsearch
    from videodeepsearch.clients.storage.elasticsearch.schema import ElasticsearchConfig
    app.state.es_ocr_client = ElasticsearchOCRClient(
        config=ElasticsearchConfig(
            host=es_cfg.host,
            port=es_cfg.port,
            user=es_cfg.user,
            password=es_cfg.password,
            use_ssl=es_cfg.use_ssl,
            verify_certs=es_cfg.verify_certs,
            index_name=es_cfg.index_name,
            request_timeout=es_cfg.request_timeout,
        )
    )
    await app.state.es_ocr_client.connect()
    logger.info(f"Elasticsearch connected: {es_cfg.url}")

    arango_cfg = settings.storage.arangodb
    arango_client = ArangoClient(hosts=arango_cfg.host)
    app.state.arango_db = arango_client.db(
        arango_cfg.database,
        username=arango_cfg.username,
        password=arango_cfg.password,
    )
    logger.info(f"ArangoDB connected: {arango_cfg.database}")

    inf_cfg = settings.inference

    app.state.qwenvl_client = QwenVLEmbeddingClient(
        config=QwenVLEmbeddingConfig(base_url=inf_cfg.qwenvl.base_url)
    )
    logger.info(f"QwenVL client initialized: {inf_cfg.qwenvl.base_url}")

    app.state.mmbert_client = MMBertClient(
        config=MMBertConfig(
            base_url=inf_cfg.mmbert.base_url,
            model_name=inf_cfg.mmbert.model_name,
        )
    )
    logger.info(f"MMBert client initialized: {inf_cfg.mmbert.base_url}")

    app.state.splade_client = SpladeClient(
        config=SpladeConfig(
            url=inf_cfg.splade.url,
            model_name=inf_cfg.splade.model_name,
            timeout=inf_cfg.splade.timeout,
            verbose=inf_cfg.splade.verbose,
            max_batch_size=inf_cfg.splade.max_batch_size,
        )
    )
    logger.info(f"SPLADE client initialized: {inf_cfg.splade.url}")

    llm_cfg = settings.llm_provider
    api_key = llm_cfg.api_key

    app.state.models = {}
    
    greeter_cfg = llm_cfg.agents.greeter
    app.state.models["greeter"] = OpenRouterResponses(
        id=greeter_cfg.model_id,
        api_key=api_key,
        base_url=llm_cfg.base_url,
    )
    
    orchestrator_cfg = llm_cfg.agents.orchestrator
    app.state.models["orchestrator"] = OpenRouterResponses(
        id=orchestrator_cfg.model_id,
        api_key=api_key,
        base_url=llm_cfg.base_url,
        parallel_tool_calls=True, 
    )
    
    planning_cfg = llm_cfg.agents.planning
    app.state.models["planning"] = OpenRouterResponses(
        id=planning_cfg.model_id,
        api_key=api_key,
        base_url=llm_cfg.base_url,
        parallel_tool_calls=True, 
    )
    
    llm_tool_cfg = llm_cfg.agents.llm_tool
    if llm_tool_cfg:
        app.state.models["llm_tool"] = OpenRouterResponses(
            id=llm_tool_cfg.model_id,
            api_key=api_key,
            base_url=llm_cfg.base_url,
        )
    else:
        app.state.models["llm_tool"] = app.state.models["planning"]
        logger.info("llm_tool model not configured, using planning model as fallback")
    
    logger.info(f"Agent models initialized: {list(app.state.models.keys())}")

    app.state.worker_models = {}
    for worker_cfg in llm_cfg.workers:
        app.state.worker_models[worker_cfg.name] = WorkerModel(
            model=OpenRouterResponses(
                id=worker_cfg.model_id,
                api_key=api_key,
                base_url=llm_cfg.base_url,
                parallel_tool_calls=True, 
            ),
            description=worker_cfg.description,
            strengths=worker_cfg.strengths,
        )
    logger.info(f"Worker models initialized: {list(app.state.worker_models.keys())}")

    logger.info("VideoDeepSearch initialized successfully")

    yield

    logger.info("Shutting down VideoDeepSearch...")

    await app.state.image_qdrant_client.close()
    await app.state.segment_qdrant_client.close()
    await app.state.audio_qdrant_client.close()
    await app.state.es_ocr_client.close()
    await app.state.qwenvl_client.close()
    await app.state.mmbert_client.close()
    app.state.splade_client.close()
    await app.state.postgres_client.close()

    logger.info("VideoDeepSearch shutdown complete")
