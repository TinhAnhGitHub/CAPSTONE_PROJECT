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


def _parse_chat_history(raw_history: list[dict]) -> list[ChatMessage]:
    print(raw_history)
    res = []
    for his in raw_history:
        message = ChatMessage.model_validate(his)
        res.append(message)
    return res
   

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
        ],
        "session_id":"123"
    }
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            user_id = data['user_id']
            list_video_ids = data['video_ids']
            user_demand = data['user_demand']
            chat_history_payload = data['chat_history']
            chat_history = _parse_chat_history(chat_history_payload)

            async_generator = workflow_service.ignite_workflow(
                user_id=user_id,
                list_video_ids=list_video_ids,
                user_demand=user_demand,
                chat_history=chat_history,
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
        print(f"Output: {output}")
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
            
    
        
