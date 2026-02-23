from app.model.session_message import (
    ThinkingStep,
)
from app.core.lifespan import app_state
from llama_index.core.base.llms.types import MessageRole
from app.api.socket_handlers.utils.session_room import session_room
import re

sio = app_state.sio

async def handle_agent_stream(session_id, data, thinking_accum, global_session_tasks):
    thinking_delta = data.get("thinking_delta", None)
    if thinking_delta:
        # also save
        title, description = parse_thinking(thinking_delta)
        thinking_step = ThinkingStep(
            title=title, description=description
        )
        thinking_accum.append(thinking_step)
        global_session_tasks[session_id][
            "thinking_accum"
        ] = thinking_accum
        await sio.emit(
            "thinking",
            {
                "title": title,
                "description": description,
            },
            to=session_room(session_id),
        )
    else:  # reponse
        response = data.get("response", "")
        response_delta = data.get("delta", "")
        # accumulate the delta
        if response_delta:
            accum += response_delta
            global_session_tasks[session_id]["accum"] = accum
        await sio.emit(
            "response",
            {
                "content": response,
                "content_delta": response_delta,
                "role": MessageRole.ASSISTANT.value,
            },
            to=session_room(session_id),
        )


def parse_thinking(text: str):
    """
    Extract **Title** and the remaining description.
    """
    match = re.match(r"\*\*(.*?)\*\*\s*(.*)", text, re.DOTALL)

    if match:
        title = match.group(1).strip()
        description = match.group(2).strip()
    else:
        title = "Thinking"
        description = text.strip()

    return title, description
