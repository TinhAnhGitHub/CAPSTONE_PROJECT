from email.mime import image
import json
from app.core.dependencies import UserServiceDep
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
                TextBlock(text=message),
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
            # lấy history giữa người dùng và bot bằng session_id
            # [{"role": "user", "content": message}, {"role": "assistant", "content": response}, ...]
            user_service = app_state.user_service
            chat_history = await user_service.get_user_chat_detail(session_id)
            chat_history_dict = []
            for chat in chat_history:
                chat_dict = chat.model_dump()
                del chat_dict["id"]
                del chat_dict["session_id"]
                del chat_dict["timestamp"]
                chat_history_dict.append(chat_dict)

            ai_url = "ws://100.113.186.28:8050/ws/start_workflow"
            payload = {
                "user_id": "string123",
                "video_ids": ["vid_e71c93a52"],
                "user_demand": message,
                "chat_history": chat_history_dict,
                # "session_id": "123",
            }

            # HTTP stream
            async with websockets.connect(ai_url) as ws:
                await ws.send(json.dumps(payload))

                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        data = data.get("data", {})
                        msg_type = data.get("event_type", "")
                        # content = data.get("content", "")
                        print("Received msg_type:", msg_type)
                        # AgentProgressEvent -> ...
                        # agentinput -> bỏ

                        # agentstream:
                        #     - thinking_delta -> hiện thinking
                        #     - thinking_delta == null && (delta = cộng dồn, reponse = all)

                        # agentoutput:
                        #     - tổng hợp những cái ở trên

                        # toolcall: -> hiện icon tool_name
                        # toolcallresult -> iserror -> icon thành công

                        # final event -> lưu chat_history

                        if (msg_type == "AgentProgressEvent"):
                            # hiện ...
                            await sio.emit(
                                "running",
                                {
                                    "content": data,
                                    "role": MessageRole.ASSISTANT.value,
                                },
                                to=socket_id,
                            )
                        elif (msg_type == "AgentInput"):
                            pass
                        elif (msg_type == "AgentStream"):
                            thinking_delta = data.get("thinking_delta", None)
                            if thinking_delta:
                                await sio.emit("thinking", {
                                    "content": thinking_delta,
                                    "role" : MessageRole.ASSISTANT.value,
                                }, to=socket_id)
                            else: # reponse
                                response = data.get("response", "")
                                response_delta = data.get("delta", "")
                                await sio.emit("response", {
                                    "content": response,
                                    "content_delta": response_delta,
                                    "role" : MessageRole.ASSISTANT.value,
                                }, to=socket_id)
                        elif (msg_type == "AgentOutput"):
                            agent_reponse = data.get("response", "")
                            agent_reponse["session_id"] = session_id
                            ai_message = SessionMessage.model_validate(agent_reponse)
                            await app_state.chat_service.add_message(session_id, "assistant", ai_message)
                            await sio.emit("full_response", {
                                "content": data,
                            })
                            # lưu vào db
                        elif (msg_type == "ToolCall"):
                            pass
                        elif (msg_type == "ToolCallResult"):
                            # show toolicon trước
                            # mốt show list hình ảnh từ s3, video + các timestamp
                            media = data.get("media", [])
                            await sio.emit("media", {
                                "content": media
                            }, to=socket_id)
                        elif (msg_type == "FinalEvent"):
                            pass
                        else:
                            pass

                    except Exception as e:
                        print("⚠️ parse error:", e)
            # update the db
            # blocks = parseFullResponseToBlocks(full_response)

        except Exception as e:
            await sio.emit(
                "error", {"message": "agent unreachable: " + str(e)}, to=socket_id
            )

        # notify finish
        # await sio.emit(
        #     "stream_end", {"timestamp": ai_message.timestamp.isoformat()}, to=socket_id
        # )

    except Exception as e:
        await sio.emit("error", {"message": str(e)}, to=socket_id)
