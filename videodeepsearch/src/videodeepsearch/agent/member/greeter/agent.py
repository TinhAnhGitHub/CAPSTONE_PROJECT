# agents/member/greeter/agent.py

from agno.agent import Agent
from agno.db.base import AsyncBaseDb, BaseDb
from agno.learn.machine import LearningMachine
from agno.learn.config import (
    UserMemoryConfig,
    UserProfileConfig,
    SessionContextConfig,
    LearningMode,
)
from agno.memory import MemoryManager
from agno.models.base import Model

def get_greeter_agent(
    session_id: str,
    user_id: str,
    model: Model,
    db: AsyncBaseDb | BaseDb,
    description: str, 
    system_prompt: str, 
    instructions: list[str]
) -> Agent:
    """
    Greeter Agent — the user-facing entry point of the system.

    Responsibilities:
    - Identify the user and recall their history across sessions.
    - Understand user intent before any delegation.
    - Delegate ALL video search/retrieval tasks to the Orchestrator Agent.
    - Store video IDs or preferences mentioned by the user in session_state.

    Memory strategy:
    - UserProfile: captures name, preferred_name, and structured user facts (ALWAYS mode).
    - UserMemory: captures unstructured observations about the user (ALWAYS mode).
    - SessionContext: tracks session goal/summary for cross-session recall (ALWAYS mode).
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
        user_profile=UserProfileConfig(
            mode=LearningMode.AGENTIC,
        ),
        user_memory=UserMemoryConfig(
            mode=LearningMode.AGENTIC,
            enable_add_memory=True,
            enable_update_memory=True,
            enable_delete_memory=False,
            enable_clear_memories=False,
        ),
        session_context=SessionContextConfig(
            mode=LearningMode.AGENTIC,
            enable_planning=False, 
        ),
    )

    return Agent(
        name="Greeter_Agent",
        role="Entry point — understand user intent and delegate to Orchestrator",
        model=model,
        user_id=user_id,
        session_id=session_id,
        db=db,

        memory_manager=memory_manager,
        enable_agentic_memory=True,
        update_memory_on_run=True,
        add_memories_to_context=True,

        # ── Session State (runtime scratchpad) ────────────────────────────
        # Stores things like: list_video_ids, last_retrieved_video_id, etc.
        add_session_state_to_context=True,
        enable_agentic_state=True,

        # ── Cross-session recall ──────────────────────────────────────────
        enable_session_summaries=True,
        add_session_summary_to_context=True,
        search_past_sessions=True,
        num_past_sessions_to_search=5,

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
        num_history_runs=5,

        debug_mode=False,
    )