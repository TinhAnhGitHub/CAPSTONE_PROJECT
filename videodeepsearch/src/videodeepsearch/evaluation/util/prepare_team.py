"""
This script will provide the API to fully prepare the team object
"""

import os
from agno.models.openrouter import OpenRouterResponses
from agno.db.postgres import AsyncPostgresDb

from videodeepsearch.agent.team import build_video_search_team
from videodeepsearch.agent.supervisor.orchestrator.spawn_toolkit import WorkerModel
from videodeepsearch.core.settings import Settings, load_settings

from videodeepsearch.clients.storage.postgre.client import PostgresClient
from videodeepsearch.clients.storage.minio.client import MinioStorageClient
from videodeepsearch.clients.storage.qdrant.image_client import ImageQdrantClient
from videodeepsearch.clients.storage.qdrant.segment_client import SegmentQdrantClient
from videodeepsearch.clients.storage.qdrant.audio_client import AudioQdrantClient
from videodeepsearch.clients.storage.elasticsearch.client import ElasticsearchOCRClient
from videodeepsearch.clients.storage.elasticsearch.schema import ElasticsearchConfig
from videodeepsearch.clients.inference.client import QwenVLEmbeddingClient, MMBertClient, SpladeClient
from videodeepsearch.clients.inference.schema import QwenVLEmbeddingConfig, MMBertConfig, SpladeConfig
from arango import ArangoClient  # type: ignore

from typing import Any
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

def _build_model_kwargs(cfg) -> dict:
    kwargs = {
        "id": cfg.model_id,
        "api_key": os.getenv("OPENROUTER_API_KEY"),
    }
    # Add optional LLM parameters if configured
    if cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature
    if cfg.top_p is not None:
        kwargs["top_p"] = cfg.top_p
    if cfg.max_output_tokens is not None:
        kwargs["max_output_tokens"] = cfg.max_output_tokens
    if cfg.max_tool_calls is not None:
        kwargs["max_tool_calls"] = cfg.max_tool_calls
    if cfg.extra_body is not None:
        kwargs["extra_body"] = cfg.extra_body
    return kwargs

def setup_models(settings: Settings) -> tuple[dict[str, OpenRouterResponses], dict[str, WorkerModel]]:
    llm_cfg = settings.llm_provider

    models: dict[str, OpenRouterResponses] = {}

    greeter_cfg = llm_cfg.agents.greeter
    models["greeter"] = OpenRouterResponses(**_build_model_kwargs(greeter_cfg))

    orchestrator_cfg = llm_cfg.agents.orchestrator
    models["orchestrator"] = OpenRouterResponses(**_build_model_kwargs(orchestrator_cfg))

    planning_cfg = llm_cfg.agents.planning
    models["planning"] = OpenRouterResponses(**_build_model_kwargs(planning_cfg))

    llm_tool_cfg = llm_cfg.agents.llm_tool
    if llm_tool_cfg:
        models["llm_tool"] = OpenRouterResponses(**_build_model_kwargs(llm_tool_cfg))
    else:
        models["llm_tool"] = models["planning"]
        logger.info("llm_tool model not configured, using planning model as fallback")

    summarizer_cfg = llm_cfg.agents.summarizer
    if summarizer_cfg:
        models["summarizer"] = OpenRouterResponses(**_build_model_kwargs(summarizer_cfg))
    else:
        models["summarizer"] = models["planning"]
        logger.info("summarizer model not configured, using planning model as fallback")

    logger.info(f"Agent models initialized: {list(models.keys())}")

    worker_models: dict[str, WorkerModel] = {}
    for worker_cfg in llm_cfg.workers:
        worker_models[worker_cfg.name] = WorkerModel(
            model=OpenRouterResponses(**_build_model_kwargs(worker_cfg)),
            description=worker_cfg.description,
            strengths=worker_cfg.strengths,
        )
    logger.info(f"Worker models initialized: {list(worker_models.keys())}")

    return models, worker_models

async def initialize_clients(settings: Settings) -> dict[str, Any]:
    clients = {}

    clients["postgres_client"] = PostgresClient(
        database_url=settings.storage.postgres.connection_url
    )
    async with clients["postgres_client"].get_session() as session:
        from sqlalchemy import text
        result = await session.execute(text("SELECT version();"))
        version = result.scalar_one()
        logger.info(f"PostgreSQL connected: {version}")

    clients["agno_db"] = AsyncPostgresDb(
        db_url=settings.storage.postgres.connection_url,
        create_schema=True,
    )
    logger.info("Agno AsyncPostgresDb initialized for session storage")

    clients["minio_client"] = MinioStorageClient(
        host=settings.storage.minio.host,
        port=settings.storage.minio.port,
        access_key=settings.storage.minio.access_key,
        secret_key=settings.storage.minio.secret_key,
        secure=settings.storage.minio.secure,
    )
    logger.info("MinIO client initialized")

    qdrant_cfg = settings.storage.qdrant
    clients["image_qdrant_client"] = ImageQdrantClient(
        host=qdrant_cfg.host,
        port=qdrant_cfg.port,
        collection_name=qdrant_cfg.collection_name,
        grpc_port=qdrant_cfg.grpc_port,
        prefer_grpc=qdrant_cfg.prefer_grpc,
    )
    clients["segment_qdrant_client"] = SegmentQdrantClient(
        host=qdrant_cfg.host,
        port=qdrant_cfg.port,
        collection_name=qdrant_cfg.collection_name,
        grpc_port=qdrant_cfg.grpc_port,
        prefer_grpc=qdrant_cfg.prefer_grpc,
    )
    clients["audio_qdrant_client"] = AudioQdrantClient(
        host=qdrant_cfg.host,
        port=qdrant_cfg.port,
        collection_name=qdrant_cfg.collection_name,
        grpc_port=qdrant_cfg.grpc_port,
        prefer_grpc=qdrant_cfg.prefer_grpc,
    )
    logger.info(f"Qdrant clients initialized: {qdrant_cfg.host}:{qdrant_cfg.port}")

    es_cfg = settings.storage.elasticsearch
    clients["es_ocr_client"] = ElasticsearchOCRClient(
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
    await clients["es_ocr_client"].connect()
    logger.info(f"Elasticsearch connected: {es_cfg.host}:{es_cfg.port}")

    # ArangoDB
    arango_cfg = settings.storage.arangodb
    arango_client = ArangoClient(hosts=arango_cfg.host)
    clients["arango_db"] = arango_client.db(
        arango_cfg.database,
        username=arango_cfg.username,
        password=arango_cfg.password,
    )
    logger.info(f"ArangoDB connected: {arango_cfg.database}")

    inf_cfg = settings.inference

    clients["qwenvl_client"] = QwenVLEmbeddingClient(
        config=QwenVLEmbeddingConfig(base_url=inf_cfg.qwenvl.base_url)
    )
    logger.info(f"QwenVL client initialized: {inf_cfg.qwenvl.base_url}")

    clients["mmbert_client"] = MMBertClient(
        config=MMBertConfig(
            base_url=inf_cfg.mmbert.base_url,
            model_name=inf_cfg.mmbert.model_name,
        )
    )
    logger.info(f"MMBert client initialized: {inf_cfg.mmbert.base_url}")

    clients["splade_client"] = SpladeClient(
        config=SpladeConfig(
            url=inf_cfg.splade.url,
            model_name=inf_cfg.splade.model_name,
            timeout=inf_cfg.splade.timeout,
            verbose=inf_cfg.splade.verbose,
            max_batch_size=inf_cfg.splade.max_batch_size,
        )
    )
    logger.info(f"SPLADE client initialized: {inf_cfg.splade.url}")

    return clients

async def cleanup_clients(clients: dict[str, Any]) -> None:
    logger.info("Cleaning up clients...")

    await clients["image_qdrant_client"].close()
    await clients["segment_qdrant_client"].close()
    await clients["audio_qdrant_client"].close()
    await clients["es_ocr_client"].close()
    await clients["qwenvl_client"].close()
    await clients["mmbert_client"].close()
    clients["splade_client"].close()
    await clients["postgres_client"].close()

    logger.info("Clients cleanup complete")
    
async def return_team(
    user_id:str,
    session_id: str,
    video_ids: list[str],
    user_demand: str,
):
    settings = load_settings()
    clients = await initialize_clients(settings)
    models, worker_models = setup_models(settings)
    
    team = build_video_search_team(
        session_id=session_id,
        user_id=user_id,
        list_video_ids=video_ids,
        models=models, #type:ignore
        worker_models=worker_models,
        db=clients["agno_db"],
        image_qdrant_client=clients["image_qdrant_client"],
        segment_qdrant_client=clients["segment_qdrant_client"],
        audio_qdrant_client=clients["audio_qdrant_client"],
        qwenvl_client=clients["qwenvl_client"],
        mmbert_client=clients["mmbert_client"],
        splade_client=clients["splade_client"],
        postgres_client=clients["postgres_client"],
        minio_client=clients["minio_client"],
        es_ocr_client=clients["es_ocr_client"],
        arango_db=clients["arango_db"],
    )
    return team