import os
from typing import Any, AsyncGenerator, Mapping

from pydantic import BaseModel
from llama_index.core.workflow import Context, JsonSerializer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ToolCallResult
from videodeepsearch.agent.agent_as_tool import (
    running_orchestrator_agent_as_tools
)
from videodeepsearch.agent.base import get_global_agent_registry
from videodeepsearch.agent.definition import GREETER_AGENT
from videodeepsearch.core.app_state import get_llm_instance
from llama_index.core.workflow import Event
from llama_index.core.agent.workflow import AgentOutput

from videodeepsearch.tools.base.registry import tool_registry, get_registry_tools

class AgentTotalOutput(Event):
    total_accumulated_events: list[Event]



DEFAULT_PHOENIX_HTTP_PORT = 6006
DEFAULT_PHOENIX_GRPC_PORT = 4317



import llama_index.core
os.environ["PHOENIX_PROJECT_NAME"] = "demo_phoenix"
llama_index.core.set_global_handler("arize_phoenix")


def _mark_context(value):
    if isinstance(value, Context): 
        return value.to_dict(serializer=JsonSerializer())
    if isinstance(value, dict): return  {k: _mark_context(v) for k, v in value.items()}
    if isinstance(value, list): return [_mark_context(item) for item in value]

    return value

def _serialize_event(event: Any):
    """Convert workflow events into JSON-serializable payloads."""
    payload: dict[str, Any]

    if isinstance(event, BaseModel):
        payload = event.model_dump()
    elif hasattr(event, "model_dump"):
        payload = event.model_dump()
    elif isinstance(event, Mapping):
        payload = dict(event)
    elif hasattr(event, "__dict__"):
        payload = event.__dict__.copy()
    else:
        payload = {"data": str(event)}

    payload.setdefault("event_type", event.__class__.__name__)
    return _mark_context(payload)



async def ignite_workflow(
    user_id: str, 
    list_video_ids:list[str], 
    user_demand: str,
    session_id: str,
    chat_history: list[ChatMessage] 
) -> AsyncGenerator[dict[str, Any], None]:

    agent_registry = get_global_agent_registry()
    

    orchestration_partial_params = {
        'session_id': session_id,
        'user_id': user_id,
        'list_video_id': list_video_ids,
        'user_original_user_message': user_demand
    }
    orchestrator_as_tools = FunctionTool.from_defaults(
        async_fn=running_orchestrator_agent_as_tools,
        partial_params=orchestration_partial_params
    )

    greeting_agent = agent_registry.spawn(
        name=GREETER_AGENT,
        llm=get_llm_instance(name=GREETER_AGENT),
        tools=[orchestrator_as_tools]
    )

    chat_history.append(
        ChatMessage(
            role=MessageRole.USER,
            content=user_demand
        )
    )

    ctx = Context(workflow=greeting_agent)

    total_events = []
    handler = greeting_agent.run(user_msg=user_demand, ctx=ctx, chat_history=chat_history)
    async for ev in handler.stream_events():


       

        if isinstance(ev, ToolCallResult):
            if ev.tool_kwargs.get('worker_tools'):
                del ev.tool_kwargs['worker_tools']

            if ev.tool_kwargs.get('orchestration_tools'):
                del ev.tool_kwargs['orchestration_tools']

            if ev.tool_kwargs.get('greeter_tools'):
                del ev.tool_kwargs['greeter_tools']

            ev.tool_output.raw_input = {}
        total_events.append(ev)
        yield _serialize_event(event=ev) # type: ignore


        # if isinstance(ev, AgentOutput):
        #     total_events_signal = AgentTotalOutput(total_accumulated_events=total_events)
        #     yield _serialize_event(event=total_events_signal) # type: ignore

    
    try:
        _ = await handler
        
    except Exception as e:
        raise e


    

    
    