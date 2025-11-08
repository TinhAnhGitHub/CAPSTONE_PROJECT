from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Annotated

from videodeepsearch.core.dependencies import get_workflow_service
from videodeepsearch.agent.orc_service import WorkflowService


router = APIRouter(
    prefix='/ws', tags=['workflow']
)



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
        "user_demand": "Summarize the videos"
    }
    """
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        user_id = data['user_id']
        list_video_ids = data['video_ids']
        user_demand = data['user_demand']
        
        async_generator = workflow_service.ignite_workflow(
            user_id=user_id,
            list_video_ids=list_video_ids,
            user_demand=user_demand
        )
        async for output in async_generator:
            await websocket.send_json(
                {
                    "type": "workflow_event",
                    "data": output
                }
            )
        await websocket.send_json({"type": "complete"})
