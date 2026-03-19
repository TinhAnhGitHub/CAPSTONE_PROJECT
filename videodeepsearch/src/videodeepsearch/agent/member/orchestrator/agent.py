# agents/member/orchestrator/agent.py
from collections.abc import Callable
from agno.agent import Agent
from agno.db.base import AsyncBaseDb, BaseDb
from agno.learn.machine import LearningMachine
from agno.learn.config import (
    SessionContextConfig,
    EntityMemoryConfig,
    LearningMode,
)
from agno.memory import MemoryManager
from agno.models.base import Model
from agno.run import RunContext
from agno.tools import Toolkit


# def get_worker_tools_factory(
#     user_id: str,
#     list_video_ids: list[str],
# ) -> Callable:
#     """
#     Returns a Callable Factory for worker tools.

#     Agno calls this function at runtime, injecting the agent instance and
#     run_context automatically. The factory builds the concrete tool list
#     from the video IDs stored in session_state (set by the Greeter).

#     Why a factory and not a static list?
#     - list_video_ids changes per request.
#     - tool_registry.get_concrete_agent_tools() needs runtime context.
#     - Agno caches the result per session when cache_callables=True.
#     """
#     def _get_worker_tools(agent: Agent, run_context: RunContext) -> list:
#         # Pull video IDs from session state if available (set by Greeter).
#         # Fall back to the IDs passed at agent construction time.
#         session_state = agent.session_state or {}
#         resolved_video_ids = session_state.get("list_video_ids", list_video_ids)

#         from videodeepsearch.tools.base.registry import tool_registry
#         return tool_registry.get_concrete_agent_tools(
#             agent_name="worker_agent",
#             user_id=user_id,
#             list_video_id=resolved_video_ids,
#         )

#     return _get_worker_tools


def get_orchestrator_agent(
    session_id: str,
    user_id: str,
    model: Model,
    db: AsyncBaseDb | BaseDb,
    spawn_agent_toolkit: Toolkit,
    description: str, 
    system_prompt: str, 
    instructions: list[str]
) -> Agent:
    """
    Orchestrator Agent — the technical execution leader.

    Responsibilities:
    - Receives the user demand from the Greeter.
    - Delegates planning to the Planning Agent (team member).
    - Dispatches Worker Agents (via Callable Factory tools) to execute each step.
    - Synthesizes worker results into a coherent final response.
    - Stores intermediate results and execution status in session_state.

    Memory strategy:
    - SessionContext with planning: tracks multi-step execution progress.
    - EntityMemory: remembers facts about video entities across sessions
      (e.g., "video_42 contains a red car at 00:02:15").
    """
    memory_manager = MemoryManager(
        model=model,
        db=db,
        add_memories=True,
        update_memories=True,
        delete_memories=False,
        clear_memories=False,
    )

    learning = LearningMachine(
        db=db,
        model=model,
        session_context=SessionContextConfig(
            mode=LearningMode.ALWAYS,
            enable_planning=True,
        ),
        entity_memory=EntityMemoryConfig(
            mode=LearningMode.ALWAYS,
            namespace="global",
            enable_create_entity=True,
            enable_add_fact=True,
            enable_add_event=True,
        ),
    )

    return Agent(
        name="Orchestrator_Agent",
        role="Coordinate planning and worker execution for video search and retrieval",
        model=model,
        user_id=user_id,
        session_id=session_id,
        db=db,

        # ── Worker tools via Callable Factory ─────────────────────────────
        # Resolved at runtime — picks up list_video_ids from session_state.
        tools=[spawn_agent_toolkit],
        cache_callables=True,

        # ── Memory ────────────────────────────────────────────────────────
        memory_manager=memory_manager,
        enable_agentic_memory=True,
        update_memory_on_run=True,
        add_memories_to_context=True,

        # ── Session State ─────────────────────────────────────────────────
        # Stores: current_step, worker_results, execution_status, etc.
        add_session_state_to_context=True,
        enable_agentic_state=True,

        # ── Session continuity ────────────────────────────────────────────
        enable_session_summaries=True,
        add_session_summary_to_context=True,

        # ── Learning Machine ──────────────────────────────────────────────
        learning=learning,
        add_learnings_to_context=True,

        # ── Context ───────────────────────────────────────────────────────
        add_datetime_to_context=True,
        markdown=True,

        # ── Prompt ────────────────────────────────────────────────────────
        description=description,
        instructions=instructions,
        system_message=system_prompt,

        # ── History ───────────────────────────────────────────────────────
        add_history_to_context=True,
        num_history_runs=3,

        # ── Retry on failure ──────────────────────────────────────────────
        retries=2,
        delay_between_retries=2,
        exponential_backoff=True,

        # ── Debug ─────────────────────────────────────────────────────────
        debug_mode=False,
    )