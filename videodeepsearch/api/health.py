from __future__ import annotations

from fastapi import APIRouter

from videodeepsearch.core.app_state import Appstate
from videodeepsearch.agent.base import get_global_agent_registry
from videodeepsearch.tools.type.registry import tool_registry


router = APIRouter(prefix="/health", tags=["health"])


def _safe_bool(value: object) -> bool:
    return bool(value)


@router.get(
    "/check_tools",
    summary="Inspect registered search tools and their metadata."
)
async def check_tools():
    all_tools = tool_registry.list_all()
    categories = {
        category: tool_registry.list_by_category(category)
        for category in tool_registry.list_all_categories()
    }
    tags = tool_registry.list_all_tags()
    return {
        "total": len(all_tools),
        "tools": all_tools,
        "categories": categories,
        "tags": tags,
    }


@router.get(
    "/check_agents",
    summary="List registered agents available in the orchestrator."
)
async def check_agents():
    registry = get_global_agent_registry()
    agents = list(registry.name2ag_conf.keys())
    return {
        "total": len(agents),
        "agents": agents,
    }


@router.get(
    "/app_state",
    summary="Check readiness of shared clients and factories stored in Appstate."
)
async def app_state_status():
    app_state = Appstate()
    return {
        "tool_factory_ready": _safe_bool(app_state.tool_factory),
        "milvus_image_connected": _safe_bool(app_state.image_milvus_client),
        "milvus_segment_connected": _safe_bool(app_state.segment_milvus_client),
        "postgres_connected": _safe_bool(app_state.postgres_client),
        "minio_connected": _safe_bool(app_state.minio_client),
        "external_client_connected": _safe_bool(app_state.external_client),
        "llm_instances": list(app_state.llm_instance.keys()),
    }


@router.get(
    "/ready",
    summary="Overall readiness signal for running the workflow."
)
async def readiness():
    state = await app_state_status()
    healthy = all(
        state[key]
        for key in (
            "tool_factory_ready",
            "milvus_image_connected",
            "milvus_segment_connected",
            "postgres_connected",
            "minio_connected",
            "external_client_connected",
        )
    )
    return {"ready": healthy, **state}

