"""
Small worker state: Future Implementation
"""
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import Event, Context
from pydantic import BaseModel, Field

class SmallWorkerContext(BaseModel):
    chat_history: list[ChatMessage] = Field(..., description="A list of chat messages that the agent generate")
    events: list[Event] = Field(..., description="Capturing all the events emitted by the agent")

async def get_small_agent_state(ctx: Context, name: str) -> SmallWorkerContext:
    current_state = ctx.store.get(name)
    return SmallWorkerContext.model_validate(current_state)

async def save_small_agent_state(ctx: Context, name: str, saved_state: SmallWorkerContext) -> None:
    async with ctx.store.edit_state() as state:
        state[name] = saved_state
    
async def get_small_worker_chat_history(ctx: Context, name: str) -> list[ChatMessage]:
    current_state = await get_small_agent_state(ctx=ctx, name=name)
    return current_state.chat_history

async def set_small_worker_chat_history(ctx:Context, name: str, chat_history: list[ChatMessage]) -> None:
    current_state = await get_small_agent_state(ctx=ctx,name=name)
    current_state.chat_history = chat_history
    await save_small_agent_state(ctx=ctx, name=name, saved_state=current_state)

async def append_small_worker_events(ctx: Context, name: str, event: Event):
    current_state = await get_small_agent_state(ctx=ctx, name=name)
    current_state.events.append(event)
    await save_small_agent_state(ctx=ctx, name=name, saved_state=current_state)