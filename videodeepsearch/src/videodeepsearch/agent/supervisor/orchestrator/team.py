"""Orchestrator Team - Handles video search and retrieval tasks."""

from typing import Any

from agno.db.base import AsyncBaseDb, BaseDb
from agno.learn.config import LearningMode, SessionContextConfig
from agno.learn.machine import LearningMachine
from agno.models.base import Model
from agno.team import Team
from agno.team.mode import TeamMode

from videodeepsearch.agent.supervisor.orchestrator.prompt import (
    ORCHESTRATOR_DESCRIPTION,
    ORCHESTRATOR_SYSTEM_PROMPT,
    ORCHESTRATOR_INSTRUCTIONS,
)
from videodeepsearch.agent.supervisor.orchestrator.spawn_toolkit import SpawnWorkerToolkit, WorkerModel
from videodeepsearch.agent.supervisor.orchestrator.metrics import aggregate_worker_metrics
from videodeepsearch.agent.member.worker.tool_selector import ToolSelector
from videodeepsearch.agent.member.worker.prompt import WORKER_SYSTEM_PROMPT
from videodeepsearch.agent.member.planning.agent import get_planning_agent
from videodeepsearch.agent.member.planning.prompt import (
    PLANNING_AGENT_DESCRIPTION,
    PLANNING_AGENT_INSTRUCTIONS,
    PLANNING_AGENT_SYSTEM_PROMPT
)
from videodeepsearch.toolkit.registry import ToolRegistry


def build_orchestrator_team(
    session_id: str,
    user_id: str,
    model: Model,
    planning_model: Model,
    worker_models: dict[str, WorkerModel],
    db: AsyncBaseDb | BaseDb,
    tool_selector: ToolSelector,
    tool_registry: ToolRegistry,
) -> Team:
    """Build the Orchestrator Team - handles video search and retrieval.

    This is a TEAM that:
    - Receives tasks from the VideoDeepSearch Team
    - Consults Planning Agent for structured execution plans
    - Spawns workers dynamically to execute plan steps
    - Synthesizes results into coherent responses

    Args:
        session_id: Session identifier
        user_id: User ID
        model: The coordinating model for this team (orchestrator model)
        planning_model: Model for the planning agent
        worker_models: Dict of WorkerModel instances for spawning workers
        db: Database instance
        tool_selector: ToolSelector for resolving tools for workers
        tool_registry: ToolRegistry for planning agent tools

    Returns:
        Team: The Orchestrator Team instance
    """
    spawn_worker_toolkit = SpawnWorkerToolkit(
        worker_models=worker_models,
        worker_instructions=[WORKER_SYSTEM_PROMPT],
        tool_selector=tool_selector,
    )

    planning_agent = get_planning_agent(
        session_id=session_id,
        user_id=user_id,
        model=planning_model,
        db=db,
        description=PLANNING_AGENT_DESCRIPTION,
        system_prompt=PLANNING_AGENT_SYSTEM_PROMPT,
        instructions=PLANNING_AGENT_INSTRUCTIONS,
        planning_toolkit=tool_registry,
    )

    return Team(
        name="Orchestrator", 
        role=ORCHESTRATOR_DESCRIPTION.strip(),
        members=[planning_agent],
        model=model,
        tools=[spawn_worker_toolkit], 
        mode=TeamMode.coordinate,
        db=db,
        user_id=user_id,
        session_id=session_id,
        post_hooks=[aggregate_worker_metrics],  # Aggregate worker metrics into session

        # learning=LearningMachine(
        #     db=db,
        #     model=model,
        #     session_context=SessionContextConfig(
        #         mode=LearningMode.ALWAYS,
        #         enable_planning=True,
        #     ),
        # ),
        # add_learnings_to_context=True,
  
        # add_session_state_to_context=False,
        # enable_agentic_state=False,
        
        # add_history_to_context=True,
        # num_history_runs=3,
        # share_member_interactions=True,  # Planning Agent sees worker results

        retries=2,
        delay_between_retries=2,
        exponential_backoff=True,

        system_message=ORCHESTRATOR_SYSTEM_PROMPT,
        markdown=True,
        instructions=ORCHESTRATOR_INSTRUCTIONS,

        stream=False,
        stream_events=True,
        store_member_responses=True, 
        debug_mode=False,
        debug_level=1
    )
