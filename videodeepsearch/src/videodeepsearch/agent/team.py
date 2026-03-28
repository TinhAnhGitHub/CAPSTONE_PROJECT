import logging
from typing import Any
from collections.abc import AsyncGenerator

from agno.db.base import AsyncBaseDb, BaseDb
from agno.models.base import Model
from agno.team import Team
from arango.database import StandardDatabase

from videodeepsearch.toolkit.registry import ToolRegistry
from videodeepsearch.toolkit.registry import get_tool_registry
from videodeepsearch.toolkit.factories import (
    make_search_factory,
    make_utility_factory,
    make_video_metadata_factory,
    make_ocr_factory,
    make_llm_factory,
    make_kg_factory,
)

from videodeepsearch.agent.supervisor.greeter import build_videodeepsearch_team
from videodeepsearch.agent.supervisor.orchestrator import build_orchestrator_team, WorkerModel
from videodeepsearch.agent.member.worker.tool_selector import ToolSelector

from videodeepsearch.clients.storage.postgre import PostgresClient
from videodeepsearch.clients.storage.minio import MinioStorageClient
from videodeepsearch.clients.storage.elasticsearch import ElasticsearchOCRClient

logger = logging.getLogger(__name__)


def _build_tool_selector(
    search_factory,
    utility_factory,
    video_metadata_factory,
    ocr_factory,
    llm_factory,
    kg_factory,
) -> ToolSelector:
    """Register all worker toolkits into a ToolSelector."""
    selector = ToolSelector()
    selector.register("search", search_factory)
    selector.register("utility", utility_factory)
    selector.register("video", video_metadata_factory)
    selector.register("ocr", ocr_factory)
    selector.register("llm", llm_factory)
    selector.register("kg", kg_factory)
    return selector


def _build_tool_registry(
    search_factory,
    utility_factory,
    video_metadata_factory,
    ocr_factory,
    llm_factory,
    kg_factory,
) -> ToolRegistry:
    registry = get_tool_registry()
    registry.register_toolkit(search_factory(), alias="VideoSearchToolkit")
    registry.register_toolkit(utility_factory(), alias="UtilityToolkit")
    registry.register_toolkit(video_metadata_factory(), alias="VideoMetadataToolkit")
    registry.register_toolkit(ocr_factory(), alias="OCRSearchToolkit")
    registry.register_toolkit(llm_factory(), alias="LLMToolkit")
    registry.register_toolkit(kg_factory(), alias="KGSearchToolkit")
    return registry


def _get_model(models: dict[str, Model], key: str) -> Model:
    """Get a model from the models dict, falling back to 'worker' if not found."""
    if key in models:
        return models[key]
    if "worker" in models:
        logger.warning(f"Model '{key}' not found, falling back to 'worker'")
        return models["worker"]
    return next(iter(models.values()))


def build_video_search_team(
    session_id: str,
    user_id: str,
    list_video_ids: list[str],
    models: dict[str, Model],
    worker_models: dict[str, WorkerModel],
    db: AsyncBaseDb | BaseDb,
    image_qdrant_client,
    segment_qdrant_client,
    audio_qdrant_client,
    qwenvl_client,
    mmbert_client,
    splade_client,
    postgres_client: PostgresClient,
    minio_client: MinioStorageClient,
    es_ocr_client: ElasticsearchOCRClient,
    arango_db: StandardDatabase,
) -> Team:
    """Assemble the VideoDeepSearch nested-team hierarchy.

    Hierarchy:
        VideoDeepSearch Team (entry point - supervisor)
        └── Orchestrator Team (supervisor)
            └── Planning Agent (member)
                └── Workers (spawned dynamically)

    Args:
        session_id: Session identifier
        user_id: User ID for context binding to tools
        list_video_ids: List of video IDs to search within
        models: Dict mapping model names to Model instances:
            - "greeter": Entry point team model
            - "orchestrator": Coordinates workers, executes plans
            - "planning": Generates execution plans
            - "llm_tool": Model for LLM tools

        worker_models: Dict mapping model names to WorkerModel instances.
    """
    greeter_model = _get_model(models, "greeter")
    orchestrator_model = _get_model(models, "orchestrator")
    planning_model = _get_model(models, "planning")
    llm_tool_model = _get_model(models, "llm_tool")

    factories = dict(
        search_factory=make_search_factory(
            image_qdrant_client, segment_qdrant_client, audio_qdrant_client,
            qwenvl_client, mmbert_client, splade_client=splade_client,
            user_id=user_id,
            video_ids=list_video_ids,
        ),
        utility_factory=make_utility_factory(postgres_client, minio_client),
        video_metadata_factory=make_video_metadata_factory(postgres_client, minio_client),
        ocr_factory=make_ocr_factory(
            es_ocr_client, mmbert_client,
            user_id=user_id,
            video_ids=list_video_ids,
        ),
        llm_factory=make_llm_factory(llm_tool_model),
        kg_factory=make_kg_factory(
            arango_db, mmbert_client,
            user_id=user_id,
            video_ids=list_video_ids,
        ),
    )

    tool_selector = _build_tool_selector(**factories)
    tool_registry = _build_tool_registry(**factories)

    orchestrator_team = build_orchestrator_team(
        session_id=session_id,
        user_id=user_id,
        model=orchestrator_model,
        planning_model=planning_model,
        worker_models=worker_models,
        db=db,
        tool_selector=tool_selector,
        tool_registry=tool_registry,
    )

    return build_videodeepsearch_team(
        session_id=session_id,
        user_id=user_id,
        model=greeter_model,
        members=[orchestrator_team],
        db=db,
    )


async def ignite_workflow(
    user_id: str,
    list_video_ids: list[str],
    user_demand: str,
    session_id: str,
    models: dict[str, Model],
    worker_models: dict[str, WorkerModel],
    db: AsyncBaseDb | BaseDb,
    image_qdrant_client,
    segment_qdrant_client,
    audio_qdrant_client,
    qwenvl_client,
    mmbert_client,
    splade_client,
    postgres_client: PostgresClient,
    minio_client: MinioStorageClient,
    es_ocr_client: ElasticsearchOCRClient,
    arango_db: StandardDatabase,
) -> AsyncGenerator[dict[str, Any], None]:
    """Main async entry point for the API layer."""
    initial_session_state: dict[str, Any] = {
        "list_video_ids": list_video_ids,
        "user_demand": user_demand,
    }

    team = build_video_search_team(
        session_id=session_id,
        user_id=user_id,
        list_video_ids=list_video_ids,
        models=models,
        worker_models=worker_models,
        db=db,
        image_qdrant_client=image_qdrant_client,
        segment_qdrant_client=segment_qdrant_client,
        audio_qdrant_client=audio_qdrant_client,
        qwenvl_client=qwenvl_client,
        mmbert_client=mmbert_client,
        splade_client=splade_client,
        postgres_client=postgres_client,
        minio_client=minio_client,
        es_ocr_client=es_ocr_client,
        arango_db=arango_db,
    )

    try:
        async for event in team.arun(
            input=user_demand,
            session_state=initial_session_state,
            stream=True,
            stream_events=True,
        ):
            yield _serialize_event(event)
    except Exception as e:
        logger.error(
            f"ignite_workflow error: session={session_id} user={user_id} — {e}",
            exc_info=True,
        )
        raise


def _serialize_event(event: Any) -> dict[str, Any]:
    from pydantic import BaseModel as PydanticModel
    if isinstance(event, PydanticModel):
        payload = event.model_dump(exclude_none=True)
    elif hasattr(event, "__dict__"):
        payload = {k: v for k, v in event.__dict__.items() if not k.startswith("_")}
    else:
        payload = {"data": str(event)}
    payload.setdefault("event_type", event.__class__.__name__)
    return payload
