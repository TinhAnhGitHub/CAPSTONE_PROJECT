from app.core.lifespan import app_state
from llama_index.core.base.llms.types import MessageRole
from app.api.socket_handlers.utils.session_room import session_room

sio = app_state.sio

async def handle_agent_progress_event(session_id, data):
    await sio.emit(
        "running",
        {
            "content": data,
            "role": MessageRole.ASSISTANT.value,
        },
        to=session_room(session_id),
    )
