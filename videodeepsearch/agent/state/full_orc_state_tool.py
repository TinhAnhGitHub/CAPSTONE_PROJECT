from typing import Callable, Awaitable, TypeVar
import functools
from llama_index.core.workflow import Context
from llama_index.core.llms import ChatMessage

from .full_orchestration import OrchestratorState, ORCHESTRATOR_STATE_KEY


T = TypeVar("T")
def with_state(auto_save: bool=True):
    def decorator(
        func: Callable[..., Awaitable[T]]
    ) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(ctx: Context, *args, **kwargs) -> T:
            state_dict = await ctx.store.get(ORCHESTRATOR_STATE_KEY)
            state = OrchestratorState.model_validate(state_dict)
            result = await func(state, *args, **kwargs)
            if auto_save:
                async with ctx.store.edit_state() as s:
                    try:
                        state.model_dump(mode="json")
                    except Exception as e:
                        raise e
                    s[ORCHESTRATOR_STATE_KEY] = state.model_dump(mode="json")
            return result
        return wrapper
    return decorator


@with_state()
async def set_add_message_to_chat_history(
    state: OrchestratorState,
    chat_messages: list[ChatMessage]
) -> None:
    state.user_chat_history.extend(chat_messages)

@with_state(auto_save=False)
async def get_chat_history(
    state: OrchestratorState,
) -> list[ChatMessage]:
    return state.user_chat_history

@with_state()
async def set_state_from_user(
    state: OrchestratorState,
    user_id,
    list_video_ids
):
    state.user_id = user_id
    state.list_video_ids = list_video_ids
    
    
@with_state(auto_save=False)
async def get_state_from_user(state: OrchestratorState):
    user_id = state.user_id
    list_video_ids = state.list_video_ids
    return user_id, list_video_ids