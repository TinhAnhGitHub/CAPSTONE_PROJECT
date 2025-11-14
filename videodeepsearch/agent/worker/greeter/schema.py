from pydantic import BaseModel, Field
from typing import Literal

class NextAgentDirective(BaseModel):
    """
    Routing decision from Greeting Agent to next agent in the multi-agent pipeline.
    
    Encapsulates the decision to answer immediately, route to planner for strategy design,
    or route to orchestrator for execution.
    """
    
    choose_next_agent: Literal['planner', 'orchestrator'] | None = Field(
        default=None,
        description=(
            "Name of the next agent to route to, or None to answer directly. "
            "Valid values: "
            "'planner' - Route to strategic planner to design multi-agent video search strategy. "
            "'orchestrator' - Route to orchestrator to execute an approved plan. "
            "None - Answer the query directly without routing (system questions, greetings, clarifications). "
            "Decision logic: None if answerable now → planner if video analysis needed → orchestrator if plan approved."
        )
    )

    reason: str = Field(
        description=(
            "Brief, specific reason for this routing decision. Should be a 1-2 sentence explanation "
            "that directly addresses what type of handling the query needs. "
            "Examples: "
            "'Query asks about video content and requires finding specific moments - needs planner strategy.' "
            "'User approved the plan and wants execution - routing to orchestrator.' "
            "'This is a greeting and can be answered immediately.' "
            "'Query is ambiguous - asking for clarification before routing to planner.'"
        )
    )

    passing_message: str = Field(
        description=(
            "Context and content to pass to the next agent (or final answer if no routing). "
            "Contents depend on routing destination: "
            "(1) If routing to 'planner': Include original user query + relevant chat history context. "
            "(2) If routing to 'orchestrator': Include the plan summary (agent names, strategies, composition) "
            "so orchestrator knows what to execute. Reference the plan details being handed off. "
            "(3) If choose_next_agent=None: Provide complete, friendly answer to user with next steps suggestions."
        )
    )
