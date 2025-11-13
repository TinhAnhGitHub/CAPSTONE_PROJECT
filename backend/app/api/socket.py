from email.mime import image
import json
from beanie import PydanticObjectId
from fastapi import HTTPException
from fastapi.security import HTTPBearer
import jwt
import socketio
import websockets
from app.schema.user import ALGORITHM, SECRET_KEY
from utils.blocks import parseFullResponseToBlocks

sio = socketio.AsyncServer(
    async_mode="asgi", cors_allowed_origins="*", logger=True, engineio_logger=True
)


import asyncio
from app.core.lifespan import app_state
from app.model.session_message import ImageBlock, SessionMessage, TextBlock, VideoBlock
from llama_index.core.base.llms.types import MessageRole
import httpx


security = HTTPBearer(auto_error=False)


# AGENT SOCKET USES RAW WS

@sio.on("stream_chat")
async def handle_stream_chat(socket_id, data: dict):
    try:
        # socket_id is the socket id to chat
        # session id is the chat bot session id
        session_id = data.get("sessionId", None)
        user_id = data.get("userId")  # could be None for guest
        message = data.get("text")
        video_ids = data.get("videos", [])  # list of video ids

        # create user message
        user_message = SessionMessage(
            session_id=None,
            role=MessageRole.USER,
            blocks=[
                TextBlock(text_content=message),
            ],
        )
        # update chat history and user message
        session_id = await app_state.chat_service.add_message(
            session_id, user_id, user_message
        )

        # tell client message received
        await sio.emit(
            "message_received",
            {
                "role": "user",
                "content": message,
                "session_id": session_id,
            },
            to=socket_id,
        )
        # send data to agent, agent stream back
        try:
            # request agent
            ai_url = "ws://100.113.186.28:8050/ws/start_workflow"
            payload = {
                "user_id": "abc123",
                "video_ids": ["video1_111", "video2_222"],
                "user_demand": "Summarize the videos",
                "chat_history": [
                    {"role": "user", "content": "previous question"},
                    {"role": "assistant", "content": "previous answer"},
                ],
                "session_id": "123",
            }

            full_response = []
            # HTTP stream
            async with websockets.connect(ai_url) as ws:
                await ws.send(json.dumps(payload))

                async for msg in ws:
                    try:
                        data = json.loads(msg)

                        # msg_type = data.get("type", "workflow_event")
                        # content = data.get("content", "")

                        await sio.emit(
                            "stream_chunk",
                            {
                                # "msg_type": msg_type,
                                "content": data,
                                "role": MessageRole.ASSISTANT.value,
                            },
                            to=socket_id,
                        )
                        full_response.append({"content": data})

                    except Exception as e:
                        print("⚠️ parse error:", e)
            # update the db
            blocks = parseFullResponseToBlocks(full_response)

            ai_message = SessionMessage(
                session_id=PydanticObjectId(session_id),
                role=MessageRole.ASSISTANT,
                blocks=blocks,
            )
            await app_state.chat_service.add_message(session_id, "ai", ai_message)
        except Exception as e:
            await sio.emit(
                "error", {"message": "agent unreachable: " + str(e)}, to=socket_id
            )

        # notify finish
        await sio.emit(
            "stream_end", {"timestamp": ai_message.timestamp.isoformat()}, to=socket_id
        )

    except Exception as e:
        await sio.emit("error", {"message": str(e)}, to=socket_id)
