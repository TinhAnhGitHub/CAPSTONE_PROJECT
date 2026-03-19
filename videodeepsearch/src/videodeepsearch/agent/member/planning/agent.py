from agno.agent import Agent
from agno.db.base import AsyncBaseDb, BaseDb
from agno.learn.machine import LearningMachine
from agno.learn.config import SessionContextConfig, LearningMode
from agno.models.base import Model
from agno.tools import Toolkit

def get_planning_agent(
    session_id: str,
    user_id: str,
    model: Model,
    db: AsyncBaseDb | BaseDb,
    description: str, 
    system_prompt: str, 
    instructions: list[str],
    planning_toolkit: Toolkit
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

    return Agent(
        name="Planning_Agent",
        role="Produce a detailed step-by-step execution plan for the given video retrieval demand",
        model=model,
        user_id=user_id,
        session_id=session_id,
        db=db,
        tools=[planning_toolkit],
        learning=learning,
        add_learnings_to_context=True,
        add_session_state_to_context=True,
        enable_agentic_state=True,
        add_datetime_to_context=True,
        markdown=True,
        description=description,
        instructions=instructions,
        system_message=system_prompt,
        debug_mode=False,
    )