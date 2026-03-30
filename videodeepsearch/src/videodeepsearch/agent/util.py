from typing import Any, Optional
from llama_index.core.tools import ToolOutput
from llama_index.core.llms import ChatMessage, MessageRole

from agno.models.response import ToolExecution
from agno.run.team import (
    RunStartedEvent,
    RunContentEvent,
    RunContentCompletedEvent,
    RunCompletedEvent,
    ToolCallStartedEvent,
    ToolCallCompletedEvent,
)

from llama_index.core.agent.workflow import (
    AgentInput,
    AgentStream,
    AgentOutput,
    ToolCall,
    ToolCallResult,
)

def convert_run_content_event_to_agent_stream(
    event: RunContentEvent,
    agent_name: str = 'agent'
):
    delta = ""
    if event.content is not None:
        delta = str(event.content)
    
    return AgentStream(
        delta=delta,
        response=delta,
        current_agent_name=agent_name,
        tool_calls=[],
        thinking_delta=event.reasoning_content,
    )

def convert_tool_call_started_event(event: ToolCallStartedEvent) -> Optional[ToolCall]:
    if event.tool is None:
        return None

    tool: ToolExecution = event.tool

    return ToolCall(
        tool_name=tool.tool_name or "",
        tool_kwargs=tool.tool_args or {},
        tool_id=tool.tool_call_id or "",
    )

def convert_tool_call_completed_event(event: ToolCallCompletedEvent) -> Optional[ToolCallResult]:
    """ToolCallCompletedEvent -> ToolCallResult"""
    if event.tool is None:
        return None

    tool: ToolExecution = event.tool
    
    print(f"{tool=}")

    content = tool.result    

    tool_output = ToolOutput(
        tool_name=tool.tool_name or "",
        content=content,
        raw_input=tool.tool_args or {},
        raw_output=content,
        is_error=False,
    )

    return ToolCallResult(
        tool_name=tool.tool_name or "",
        tool_kwargs=tool.tool_args or {},
        tool_id=tool.tool_call_id or "",
        tool_output=tool_output,
        return_direct=False,
    )

def convert_run_completed_event(event: RunCompletedEvent, agent_name: str = "agent") -> AgentOutput:
    
    print(f"Event in completed: {event=}")
    content = ""
    if event.content is not None:
        content = str(event.content)

    response_message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content=content,
    )

    return AgentOutput(
        response=response_message,
        current_agent_name=agent_name,
        tool_calls=[],
        raw=event,
    )