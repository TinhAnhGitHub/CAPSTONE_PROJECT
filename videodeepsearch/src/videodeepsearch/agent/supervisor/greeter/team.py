from typing import Any

from agno.db.base import AsyncBaseDb, BaseDb
from agno.memory import MemoryManager
from agno.models.base import Model
from agno.team import Team
from agno.team.mode import TeamMode

from videodeepsearch.agent.supervisor.greeter.prompt import (
    GREETER_VIDEO_DEEPSEARCH_DESCRIPTION,
    GREETER_VIDEO_DEEPSEARCH_SYSTEM_PROMPT,
    GREETER_VIDEO_DEEPSEARCH_INSTRUCTIONS,
)


def build_videodeepsearch_team(
    session_id: str,
    user_id: str,
    model: Model,
    members: list,  
    db: AsyncBaseDb | BaseDb,
) -> Team:
    """Build the VideoDeepSearch Team - the main entry point for video search.

    This is a TEAM that coordinates between handling simple queries directly
    and delegating video search tasks to its members (typically the orchestrator).

    Args:
        session_id: Session identifier
        user_id: User ID for context binding
        model: The coordinating model for this team
        members: List of Team/Agent members to delegate to (e.g., orchestrator_subteam)
        db: Database for session state and memory

    Returns:
        Team: The VideoDeepSearch Team instance
    """
    memory_manager = MemoryManager(
        model=model,
        db=db,
        add_memories=True,
        update_memories=True,
        delete_memories=False,
        clear_memories=False,
    )

    return Team(
        name="VideoDeepSearch",
        role=GREETER_VIDEO_DEEPSEARCH_DESCRIPTION.strip(),
        members=members,
        model=model,
        mode=TeamMode.coordinate,
        db=db,
        user_id=user_id,
        session_id=session_id,

        get_member_information_tool=True,
        add_member_tools_to_context=True,
        determine_input_for_members=True,
        add_session_state_to_context=False,
        enable_agentic_state=False,

        memory_manager=memory_manager,
        enable_agentic_memory=True,
        update_memory_on_run=True,
        add_memories_to_context=True,

        enable_session_summaries=True,
        add_session_summary_to_context=True,

        add_history_to_context=True,
        num_history_runs=5,
        read_chat_history=True,

        retries=2,
        delay_between_retries=2,
        exponential_backoff=True,

        system_message=GREETER_VIDEO_DEEPSEARCH_SYSTEM_PROMPT,
        markdown=True,
        instructions=GREETER_VIDEO_DEEPSEARCH_INSTRUCTIONS,
        
        stream=False,
        stream_events=True,
        debug_mode=False,
        debug_level=1
    )
