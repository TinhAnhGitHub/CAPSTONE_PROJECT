import logging
from typing import Any, AsyncGenerator

from agno.db.base import AsyncBaseDb, BaseDb
from agno.learn.config import LearningMode, SessionContextConfig
from agno.learn.machine import LearningMachine
from agno.models.base import Model
from agno.team import Team
from agno.team.mode import TeamMode
from arango.database import StandardDatabase

from videodeepsearch.toolkit.registry import ToolRegistry, get_tool_registry
from videodeepsearch.toolkit.factories import (
    make_search_factory,
    make_utility_factory,
    make_video_metadata_factory,
    make_ocr_factory,
    make_llm_factory,
    make_kg_factory,
)

from videodeepsearch.agent.member.greeter.agent import get_greeter_agent
from videodeepsearch.agent.member.greeter.prompt import (
    GREETER_DESCRIPTION,
    GREETER_INSTRUCTIONS,
    GREETER_SYSTEM_PROMPT,
)

from videodeepsearch.agent.member.orchestrator.agent import get_orchestrator_agent
from videodeepsearch.agent.member.orchestrator.prompt import (
    ORCHESTRATOR_DESCRIPTION,
    ORCHESTRATOR_INSTRUCTIONS,
)

from videodeepsearch.agent.member.planning.agent import get_planning_agent
from videodeepsearch.agent.member.planning.prompt import (
    PLANNING_AGENT_DESCRIPTION,
    PLANNING_AGENT_INSTRUCTIONS,
    PLANNING_AGENT_SYSTEM_PROMPT
)

from videodeepsearch.agent.member.worker.spawn_toolkit import SpawnWorkerToolkit, WorkerModel
from videodeepsearch.agent.member.worker.tool_selector import ToolSelector
from videodeepsearch.agent.member.worker.prompt import WORKER_SYSTEM_PROMPT

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
    """Populate the ToolRegistry for the PlanningContextToolkit."""
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
        Greeter (Big Boss)
        └── Orchestrator Sub-Team
            ├── Orchestrator Agent
            └── Planning Agent
                └── Workers (spawned dynamically)

    Args:
        models: Dict mapping model names to Model instances:
            - "greeter": Lead agent - coordinates team, talks to user
            - "orchestrator": Coordinates workers, executes plans
            - "planning": Generates execution plans

        worker_models: Dict mapping model names to WorkerModel instances.
            Each WorkerModel includes: model, description, strengths.
    """
    greeter_model = _get_model(models, "greeter")
    orchestrator_model = _get_model(models, "orchestrator")
    planning_model = _get_model(models, "planning")

    factories = dict(
        search=make_search_factory(
            image_qdrant_client, segment_qdrant_client, audio_qdrant_client,
            qwenvl_client, mmbert_client, splade_client,
        ),
        utility=make_utility_factory(postgres_client, minio_client),
        video=make_video_metadata_factory(postgres_client, minio_client),
        ocr=make_ocr_factory(es_ocr_client, mmbert_client),
        llm=make_llm_factory(orchestrator_model),
        kg=make_kg_factory(arango_db, mmbert_client),
    )

    tool_selector = _build_tool_selector(**factories)
    tool_registry = _build_tool_registry(**factories)

    spawn_worker_toolkit = SpawnWorkerToolkit(
        worker_models=worker_models,
        worker_instructions=[WORKER_SYSTEM_PROMPT],
        tool_selector=tool_selector,
    )

    planning_agent = get_planning_agent(
        session_id=session_id, user_id=user_id, model=planning_model, db=db,
        description=PLANNING_AGENT_DESCRIPTION,
        system_prompt=PLANNING_AGENT_SYSTEM_PROMPT,
        instructions=[PLANNING_AGENT_INSTRUCTIONS],
        planning_toolkit=tool_registry,
    )

    orchestrator_agent = get_orchestrator_agent(
        session_id=session_id, user_id=user_id, model=orchestrator_model, db=db,
        spawn_agent_toolkit=spawn_worker_toolkit,
        description=ORCHESTRATOR_DESCRIPTION,
        system_prompt="",
        instructions=[ORCHESTRATOR_INSTRUCTIONS],
    )

    orchestrator_subteam = Team(
        name="Orchestrator_SubTeam",
        role="Handle all video search and retrieval tasks delegated by the Greeter",
        members=[orchestrator_agent, planning_agent],
        model=orchestrator_model,
        mode=TeamMode.coordinate,
        db=db,
        user_id=user_id,
        session_id=session_id,
        learning=LearningMachine(
            db=db,
            model=orchestrator_model,
            session_context=SessionContextConfig(
                mode=LearningMode.ALWAYS,
                enable_planning=True,
            ),
        ),
        add_learnings_to_context=True,
        add_session_state_to_context=True,
        enable_agentic_state=True,
        add_history_to_context=True,
        num_history_runs=3,
        markdown=True,
        instructions=[
            "The Orchestrator Agent leads this sub-team.",
            "Call get_available_models() to see which models you can assign to workers.",
            "Call get_available_worker_tools() to see all available tool names.",
            "Always consult the Planning Agent FIRST to get a structured plan.",
            "In the plan, each step should specify: tool_names, model_name.",
            "Workers are spawned via spawn_and_run_worker() — NOT as team members.",
            "Store intermediate results in session_state after each worker completes.",
            "Synthesise all worker results into a single coherent response.",
        ],
        stream=True,
        stream_events=True,
        debug_mode=False,
    )

    greeter_agent = get_greeter_agent(
       session_id=session_id, user_id=user_id, model=greeter_model, db=db,
       description=GREETER_DESCRIPTION,
        system_prompt=GREETER_SYSTEM_PROMPT,
       instructions=[GREETER_INSTRUCTIONS],
   )

    return Team(
        name="VideoDeepSearch_Team",
        role="VideoDeepSearch — AI-powered multi-modal video retrieval system",
        members=[greeter_agent, orchestrator_subteam],
        model=greeter_model,  
        mode=TeamMode.coordinate,
        db=db,
        user_id=user_id,
        session_id=session_id,
        add_session_state_to_context=True,
        markdown=True,
        instructions=[
            "Greeter is the LEAD agent - it coordinates the entire team.",
            "Greeter receives user queries and delegates to Orchestrator Sub-Team.",
            "Greeter is the ONLY agent that communicates directly with the user.",
            "The Orchestrator Sub-Team handles all technical video retrieval tasks.",
            "Never expose internal agent names, tool names, or implementation details.",
        ],
        stream=True,
        stream_events=True,
        debug_mode=False,
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
    """Main async entry point for the API layer.

    Args:
        models: Models for agents (greeter, orchestrator, planning)
        worker_models: Models for workers with descriptions for planning agent selection
    """
    initial_session_state: dict[str, Any] = {
        "list_video_ids": list_video_ids,
        "user_demand": user_demand,
    }

    team = build_video_search_team(
        session_id=session_id,
        user_id=user_id,
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
