from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
from typing import Any
from videodeepsearch.agent.team import ignite_workflow


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/ws', tags=['workflow']
)


@router.websocket("/start_workflow")
async def start_workflow_ws(
    websocket: WebSocket,
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

            required_fields = ['user_id', 'video_ids', 'user_demand', 'session_id']
            missing = [f for f in required_fields if f not in data]
            if missing:
                await websocket.send_json({
                    'type': 'error',
                    'error': f'Missing required fields: {missing}'
                })
                return

            user_id = data['user_id']
            list_video_ids = data['video_ids']
            user_demand = data['user_demand']
            session_id = data['session_id']

            state = websocket.app.state

            async_generator = ignite_workflow(
                user_id=user_id,
                list_video_ids=list_video_ids,
                user_demand=user_demand,
                session_id=session_id,
                models=state.models,
                worker_models=state.worker_models,
                db=state.agno_db,
                image_qdrant_client=state.image_qdrant_client,
                segment_qdrant_client=state.segment_qdrant_client,
                audio_qdrant_client=state.audio_qdrant_client,
                qwenvl_client=state.qwenvl_client,
                mmbert_client=state.mmbert_client,
                splade_client=state.splade_client,
                postgres_client=state.postgres_client,
                minio_client=state.minio_client,
                es_ocr_client=state.es_ocr_client,
                arango_db=state.arango_db,
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
        logger.exception("Unhandled error in start_workflow_ws")

        try:
            await websocket.send_json({
                'type': "error",
                'error': "An internal error occurred. Please try again.",
            })
        except Exception:
            pass
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
        return
