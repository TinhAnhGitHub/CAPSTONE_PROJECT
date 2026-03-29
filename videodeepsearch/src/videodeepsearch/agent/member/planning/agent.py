from agno.agent import Agent
from agno.db.base import AsyncBaseDb, BaseDb
from agno.learn.machine import LearningMachine
from agno.learn.config import SessionContextConfig, LearningMode
from agno.models.base import Model
from agno.tools import Toolkit

from videodeepsearch.toolkit.registry import ToolRegistry


def get_planning_agent(
    session_id: str,
    user_id: str,
    model: Model,
    db: AsyncBaseDb | BaseDb,
    description: str,
    system_prompt: str,
    instructions: list[str],
    planning_toolkit: Toolkit | ToolRegistry,
) -> Agent:
    """
    Planning Agent — a member of the Orchestrator sub-team.

    Responsibilities:
    - Receive the user demand from the Orchestrator.
    - Produce a detailed, ordered, step-by-step execution plan.
    - Learn from past plans to improve plan quality over time.

    Memory strategy:
    - SessionContext with planning mode: tracks goal, plan, and progress
      within the current session so the plan adapts as execution proceeds.
    """
    learning = LearningMachine(
        db=db,
        model=model,
        session_context=SessionContextConfig(
            mode=LearningMode.ALWAYS,
            enable_planning=True,
        ),
    )

    if isinstance(planning_toolkit, ToolRegistry):
        tool_context = planning_toolkit.generate_planning_context()
        enhanced_instructions = instructions + [tool_context]
        tools_list = []
    else:
        enhanced_instructions = instructions
        tools_list = [planning_toolkit]

    return Agent(
        name="Planning_Agent",
        role="Produce a detailed step-by-step execution plan for the given video retrieval demand",
        model=model,
        user_id=user_id,
        session_id=session_id,
        db=db,
        tools=tools_list,
        learning=learning,
        # Session state inherited from parent teams
        add_learnings_to_context=True,
        # Nested agents must NOT have enable_agentic_state=True
        add_session_state_to_context=False,
        enable_agentic_state=False,
        add_datetime_to_context=True,
        markdown=True,
        description=description,
        instructions=enhanced_instructions,
        system_message=system_prompt,
        debug_mode=False,
        debug_level=1,
        stream_events=True,
        stream=False,
    )