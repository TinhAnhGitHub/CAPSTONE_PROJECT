"""
Test script for agent versioning.

Run this to verify that MLflow LoggedModel versioning works correctly.

Usage:
    cd videodeepsearch
    uv run python test_versioning.py
"""

import asyncio
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from videodeepsearch.tracing import (
    setup_versioned_agent,
    get_version_model_id,
    get_git_info,
    compute_agent_code_hash,
    get_logged_model_info,
)
from videodeepsearch.core.settings import load_settings
from videodeepsearch.agent.team import build_video_search_team
from videodeepsearch.clients.storage.postgre.client import PostgresClient
from videodeepsearch.clients.storage.minio.client import MinioStorageClient
from videodeepsearch.clients.storage.qdrant.image_client import ImageQdrantClient
from videodeepsearch.clients.storage.qdrant.segment_client import SegmentQdrantClient
from videodeepsearch.clients.storage.qdrant.audio_client import AudioQdrantClient
from videodeepsearch.clients.storage.elasticsearch.client import ElasticsearchOCRClient
from videodeepsearch.clients.storage.elasticsearch.schema import ElasticsearchConfig
from videodeepsearch.clients.inference.client import QwenVLEmbeddingClient, MMBertClient, SpladeClient
from videodeepsearch.clients.inference.schema import QwenVLEmbeddingConfig, MMBertConfig, SpladeConfig

from agno.db.postgres import AsyncPostgresDb
from agno.models.openrouter import OpenRouterResponses
from arango import ArangoClient

load_dotenv()


def setup_models(settings):
    llm_cfg = settings.llm_provider
    models = {}

    greeter_cfg = llm_cfg.agents.greeter
    models["greeter"] = OpenRouterResponses(
        id=greeter_cfg.model_id,
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    orchestrator_cfg = llm_cfg.agents.orchestrator
    models["orchestrator"] = OpenRouterResponses(
        id=orchestrator_cfg.model_id,
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    planning_cfg = llm_cfg.agents.planning
    models["planning"] = OpenRouterResponses(
        id=planning_cfg.model_id,
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    models["llm_tool"] = models["planning"]
    models["summarizer"] = models["planning"]

    return models


async def init_clients(settings):
    clients = {}

    clients["postgres_client"] = PostgresClient(
        database_url=settings.storage.postgres.connection_url
    )

    clients["agno_db"] = AsyncPostgresDb(
        db_url=settings.storage.postgres.connection_url,
        create_schema=True,
    )

    clients["minio_client"] = MinioStorageClient(
        host=settings.storage.minio.host,
        port=settings.storage.minio.port,
        access_key=settings.storage.minio.access_key,
        secret_key=settings.storage.minio.secret_key,
        secure=settings.storage.minio.secure,
    )

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

    arango_cfg = settings.storage.arangodb
    arango_client = ArangoClient(hosts=arango_cfg.host)
    clients["arango_db"] = arango_client.db(
        arango_cfg.database,
        username=arango_cfg.username,
        password=arango_cfg.password,
    )

    inf_cfg = settings.inference
    clients["qwenvl_client"] = QwenVLEmbeddingClient(
        config=QwenVLEmbeddingConfig(base_url=inf_cfg.qwenvl.base_url)
    )
    clients["mmbert_client"] = MMBertClient(
        config=MMBertConfig(
            base_url=inf_cfg.mmbert.base_url,
            model_name=inf_cfg.mmbert.model_name,
        )
    )
    clients["splade_client"] = SpladeClient(
        config=SpladeConfig(
            url=inf_cfg.splade.url,
            model_name=inf_cfg.splade.model_name,
            timeout=inf_cfg.splade.timeout,
            verbose=inf_cfg.splade.verbose,
            max_batch_size=inf_cfg.splade.max_batch_size,
        )
    )

    return clients


async def test_versioning():
    """Test the agent versioning workflow."""

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://100.113.186.28:5000"))
    mlflow.set_experiment("agent-versioning-test")

    logger.info("=" * 60)
    logger.info("Testing Agent Versioning")
    logger.info("=" * 60)

    git_info = get_git_info()
    logger.info(f"Git info: {git_info}")
    logger.info(f"Agent code hash: {compute_agent_code_hash()}")

    settings = load_settings()
    logger.info("Settings loaded")

    clients = await init_clients(settings)
    logger.info("Clients initialized")

    models = setup_models(settings)
    logger.info(f"Models: {list(models.keys())}")

    team = build_video_search_team(
        session_id="test-versioning-session",
        user_id="test-user",
        list_video_ids=["test-video-1"],
        models=models,
        worker_models={},
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
    logger.info(f"Team built: {team.name}")

    logger.info("=" * 60)
    logger.info("Setting up versioned agent...")
    logger.info("=" * 60)

    version_ctx = setup_versioned_agent(
        team=team,
        settings=settings,
        agent_name="video-search-team",
        version_tag="test-v1",  # Optional tag
        log_config=True,
    )

    logger.info(f"Version context: {version_ctx}")
    logger.info(f"Model ID: {version_ctx.model_id}")
    logger.info(f"Version name: {version_ctx.version_name}")

    logger.info("=" * 60)
    logger.info("Verifying logged model info...")
    logger.info("=" * 60)

    model_info = get_logged_model_info(version_ctx.model_id)
    logger.info(f"LoggedModel info: {model_info}")

    
    logger.info("=" * 60)
    logger.info("Running a simple test query...")
    logger.info("=" * 60)

    user_demand = "Hello, can you help me find information about a video?"

    # with mlflow.start_run(run_name=f"test-{version_ctx.version_name}"):
    #     mlflow.log_param("user_demand", user_demand)
    #     mlflow.log_param("version_name", version_ctx.version_name)
    #     mlflow.log_param("model_id", version_ctx.model_id)

        
    #     result = await team.arun(
    #         input=user_demand,
    #         session_state={"list_video_ids": ["test-video-1"], "user_demand": user_demand},
    #         stream=False,
    #     )

    #     logger.info(f"Team response: {result.content if hasattr(result, 'content') else result}")

    #     # Get session metrics
    #     session_metrics = await team.aget_session_metrics()
    #     logger.info(f"Session metrics: input={session_metrics.input_tokens}, output={session_metrics.output_tokens}")

    # === FINAL SUMMARY ===
    logger.success("=" * 60)
    logger.success("Agent Versioning Test Complete!")
    logger.success("=" * 60)
    logger.success(f"Version name: {version_ctx.version_name}")
    logger.success(f"Model ID: {version_ctx.model_id}")
    logger.success(f"View at: {os.getenv('MLFLOW_TRACKING_URI', 'http://100.113.186.28:5000')}")
    logger.success(f"Experiment: agent-versioning-test")
    logger.success("")
    logger.success("Check MLflow UI:")
    logger.success("  - Models tab: See the LoggedModel with git info")
    logger.success("  - Traces tab: See traces linked to this version")


if __name__ == "__main__":
    asyncio.run(test_versioning())