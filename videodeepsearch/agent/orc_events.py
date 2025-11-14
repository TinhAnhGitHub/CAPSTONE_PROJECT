"""
Custom Events for Multi-Agent Video Deep Search Orchestration

This module defines custom events specific to the video deep search workflow orchestration.
These events are designed for UI emission and workflow coordination, separate from
LlamaIndex framework events.
"""
from typing import Any, Callable, Coroutine
from pydantic import Field
from llama_index.core.workflow import StartEvent, StopEvent, Event
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool


class UserInputEvent(StartEvent):
    user_id: str = Field(..., description="Unique identifier for the user")
    list_video_ids: list[str] = Field(
        default_factory=list, 
        description="List of video IDs to analyze"
    )
    user_demand: str = Field(..., description="User's natural language request")
    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation context"
    )

    llama_index_func_tools: dict[str, FunctionTool] = Field(
        default_factory=dict,
        description="LlamaIndex function tools available for agents"
    )
    normal_func: dict[str, Callable[..., Any]] = Field(
        default_factory=dict,
        description="Standard Python functions available"
    )

class FinalResponseEvent(Event):
    """
    Final consolidated response ready for user presentation.
    
    UI Emission: Display final answer, hide loading states
    Stage: Completion
    
    This event aggregates all agent outputs and produces the final
    user-facing response after all processing is complete.
    """
    passing_messages: list[ChatMessage] = Field(..., description="List of structured message objects or data payloads from the preceding agents.")

# ============================================================================
# PLANNING & ORCHESTRATION EVENTS
# ============================================================================
class   PlannerInputEvent(Event):
    """
    Input to the planning agent for strategy generation.
    
    UI Emission: Show "Planning in progress..." indicator
    Stage: Planning
    """
    user_msg: str = Field(..., description="Original user message")
    planner_demand: str = Field(
        ..., 
        description="Specific planning requirements from greeter agent"
    )

class PlanProposedEvent(Event):
    """
    Plan has been generated and is ready for review/execution.
    
    UI Emission: Display proposed plan, optionally wait for approval
    Stage: Plan Review
    
    In Human-in-the-Loop (HIL) mode, workflow pauses here for user approval.
    """
    user_msg: str = Field(..., description="Original user request")
    agent_response: str = Field(
        ..., 
        description="Planner's response and reasoning"
    )


class AgentProgressEvent(Event):
    """
    Real-time progress update from an agent.
    
    UI Emission: Update progress bar, show current agent activity
    Stage: Any (during agent execution)
    
    Use this for granular progress tracking of individual agents.
    """
    agent_name: str = Field(..., description="Name of the reporting agent")
    answer: Any = Field(
        ..., 
        description="Progress data (can be string, dict, or structured output)"
    )

class PlanningAgentEvent(Event):
    reason: str
    plan_detail: list[dict]
    plan_summary: str  


class FinalEvent(StopEvent):
    workflow_response: str
    chat_history: list[ChatMessage]


class AgentDecision(Event):
    name:str
    decision:str
    reason:str