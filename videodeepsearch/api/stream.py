from fastapi import APIRouter, WebSocket, Depends, WebSocketDisconnect, WebSocketException
from typing import Annotated, Iterable
import traceback
import logging
from llama_index.core.llms import ChatMessage
from typing import Any

from videodeepsearch.core.dependencies import get_workflow_service
from videodeepsearch.agent.orc_service import WorkflowService


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/ws', tags=['workflow']
)


def _parse_chat_history(raw_history: Iterable[dict] | None) -> list[ChatMessage]:
    """Convert JSON payload chat history into ChatMessage objects."""
    chat_messages: list[ChatMessage] = []
    if not raw_history:
        return chat_messages

    for entry in raw_history:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role") or "user"
        content = entry.get("content")
        if content is None:
            continue
        chat_messages.append(ChatMessage(role=role, content=content))
    return chat_messages


def _json_safe(obj: Any, _seen: set[int] | None = None) -> Any:
    if _seen is None:
        _seen = set()
    
    object_id = id(obj)
    if object_id in _seen:
        return "<circular>"

    _seen.add(object_id)

    if obj is None or isinstance(obj, (int,str,bool,float)):
        return obj
    
    if isinstance(obj, dict):
        safe_dict = {}
        for k,v in obj.items():
            try:
                sk = str(k)
            except Exception:
                sk = "<non-string-key>"
            
            if sk in {"parent_ctx", "tool_factory", "code_execute_env"}:
                safe_dict[sk] = '<omitted>'
                continue
            

@router.websocket("/start_workflow")
async def start_workflow_ws(
    websocket: WebSocket,
    workflow_service: Annotated[WorkflowService, Depends(get_workflow_service)]
):
    """
    WebSocket endpoint for streaming workflow results.
    Expected JSON message:
    {
        "user_id": "abc123",
        "video_ids": ["v1", "v2"],
        "user_demand": "Summarize the videos",
        "chat_history": [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"}
        ]
    }
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            user_id = data['user_id']
            list_video_ids = data['video_ids']
            user_demand = data['user_demand']
            chat_history_payload = data.get('chat_history')
            session_id = data['session_id']
            chat_history = _parse_chat_history(chat_history_payload)

            async_generator = workflow_service.ignite_workflow(
                user_id=user_id,
                list_video_ids=list_video_ids,
                user_demand=user_demand,
                chat_history=chat_history,
                session_id=session_id
            )

            async for output in async_generator:
                await websocket.send_json(
                    {
                        "type": "workflow_event",
                        "data": output
                    }
                )

            await websocket.send_json({"type": "complete"})
            return

    except WebSocketDisconnect as e:
        code = getattr(e, "code", None)
        reason = getattr(e, "reason", "")
        logger.info("WebSocket disconnected code=%s reason=%s", code, reason)
        try:
            await websocket.close(code=code or 1000)
        except Exception:
            pass
        return

    except Exception as e:
        # Log full traceback to server logs for debugging visibility
        
        print(f"Object cause error: {output}") #type:ignore
        logger.exception("Unhandled error in start_workflow_ws")
        tb = traceback.format_exc()
        
        try:
            await websocket.send_json({
                'type': "error",
                'error': str(e),
                'traceback': tb,
            })
        except Exception:
            pass
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
        return
            
    
        
