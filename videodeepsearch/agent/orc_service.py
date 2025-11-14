from typing import Any, AsyncGenerator, Mapping
from datetime import datetime
from llama_index.core.workflow import Context, JsonSerializer
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel
from pathlib import Path

from .orc_events import UserInputEvent
from .workflow import VideoAgentWorkFlow

from videodeepsearch.core.app_state import Appstate
from videodeepsearch.tools.schema.artifact import ARTIFACT_MODELS 
from videodeepsearch.agent.state import create_orchestrator_initial_state, ORCHESTRATOR_STATE_KEY
from videodeepsearch.agent.orc_events import FinalEvent
from videodeepsearch.agent.state.state_management import SessionKey, SimpleContextManager


BASEDIR = "/home/tinhanhnguyen/Desktop/HK7/Capstone/CAPSTONE_PROJECT/videodeepsearch/test/local2"
COUNTER = 0


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


class WorkflowService:
    def __init__(
        self,
        orchestration: VideoAgentWorkFlow
    ):
        self.orchestration = orchestration
    
    async def ignite_workflow(
            self, 
            user_id: str, 
            list_video_ids:list[str], 
            session_id: str,
            user_demand: str,
            chat_history: list[ChatMessage] | None = None

        ) -> AsyncGenerator[dict[str, Any], None]:
        global COUNTER
        tool_factory = Appstate().tool_factory

        get_all_llamaindex_functools = tool_factory.get_all_tools_functool(
            user_id=user_id,
            list_video_id=list_video_ids,
            agent_bucket=f'agent_test_ith_{COUNTER}_{datetime.now().strftime("%Y%m%d%H%M%S")}',
            agent_object_folder='agent_worker_small'
        )
        get_all_code_normal_tools = tool_factory.get_all_tools_normal(
            user_id=user_id,
            list_video_id=list_video_ids,
            agent_bucket=f'agent_test_ith_{COUNTER}_{datetime.now().strftime("%Y%m%d%H%M%S")}',
            agent_object_folder='agent_worker_small'
        )
        global_dependencies = tool_factory.dependency_map
        global_dependencies.update(
            {type_.__name__: type_ for type_ in ARTIFACT_MODELS}
        )
        Appstate().code_sandbox._globals.update(get_all_code_normal_tools) # type: ignore[arg-type]
        
        COUNTER += 1
    
        start_event = UserInputEvent(
            user_id=user_id,
            list_video_ids=list_video_ids,
            chat_history=chat_history or [],
            user_demand=user_demand,
            llama_index_func_tools=get_all_llamaindex_functools,
            normal_func=get_all_code_normal_tools,
        )
        session_key = SessionKey(user_id=user_id, session_id=session_id)
        context_manager = SimpleContextManager(base_dir=Path(BASEDIR))
        try:
            ctx = context_manager.load(key=session_key, workflow=self.orchestration)
        except Exception as e:
            print(e)
            print("No available context, creating new context")
            ctx = Context(self.orchestration)
            global_orchestrate_state = create_orchestrator_initial_state()
            async with ctx.store.edit_state() as state:
                state[ORCHESTRATOR_STATE_KEY] = global_orchestrate_state
            
        handler = self.orchestration.run(
            start_event=start_event,
            ctx=ctx
        )
        async for ev in handler.stream_events():
            yield _serialize_event(ev) # type: ignore
        try:
            result = await handler
            if result is not None:
                yield _serialize_event(result) # type: ignore
        except Exception as e:
            raise e
        
        context_manager.save(key=session_key, ctx=ctx)
