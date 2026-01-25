from collections import defaultdict
from email.mime import image
import json
import traceback

from regex import P
from app.core.dependencies import UserServiceDep
from beanie import PydanticObjectId
from fastapi import HTTPException
from fastapi.security import HTTPBearer
import jwt
import socketio
import websockets
from app.schema.user import ALGORITHM, SECRET_KEY
from utils.blocks import parseFullResponseToBlocks
from llama_index.core.agent.workflow import AgentOutput

sio = socketio.AsyncServer(
    async_mode="asgi", cors_allowed_origins="*", logger=True, engineio_logger=True
)


import asyncio
from app.core.lifespan import app_state
from app.model.session_message import (
    ImageBlock,
    SessionMessage,
    TextBlock,
    ThinkingBlock,
    ThinkingStep,
    ToolStep,
    ToolsBlock,
    VideoBlock,
    VideoSegment,
)

from llama_index.core.base.llms.types import MessageRole
import httpx


security = HTTPBearer(auto_error=False)

from app.tests.test_data import image_search_result_1

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
            # payload = {
            #     "user_id": user_id,
            #     "video_ids": video_ids,
            #     "user_demand": message,
            #     "chat_history": chat_history_dict,
            # }
            payload = {
                "session_id": session_id,
                "user_id": user_id,
                "video_ids": [
                    "692ad412086ada3a309334ff",
                    "692ad412086ada3a30933500",
                    "692ad412086ada3a30933501",
                    "692ad412086ada3a30933503",
                ],
                "user_demand": message,
                "chat_history": [
                    {
                        "role": "user",
                        "content": "I want to find a moment related to Tokyo city, where they develop an underground drainage system to cope with climate change. Could you find it for me?????",
                    }
                ],
            }

            # HTTP stream
            async with websockets.connect(ai_url) as ws:
                await ws.send(json.dumps(payload))

                accum = ""
                prev_msg_type = ""
                tools_accum = []
                thinking_accum = []
                ai_message_blocks = []
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        data = data.get("data", {})
                        msg_type = data.get("event_type", "")

                        # Save accumulated stream when transitioning away from AgentStream
                        if prev_msg_type == "AgentStream" and msg_type != "AgentStream":
                            if accum:
                                ai_message_block = TextBlock(text=accum)
                                ai_message_blocks.append(ai_message_block)
                                accum = ""
                            elif thinking_accum:
                                thinking_message_block = ThinkingBlock(steps=thinking_accum)
                                ai_message_blocks.append(thinking_message_block)
                                thinking_accum = []

                        if (
                            prev_msg_type == "ToolCallResult"
                            or prev_msg_type == "ToolCall"
                        ) and (msg_type != "ToolCallResult" and msg_type != "ToolCall"):
                            if tools_accum:
                                tool_message_block = ToolsBlock(steps=tools_accum)
                                ai_message_blocks.append(tool_message_block)
                                tools_accum = []
                        # content = data.get("content", "")
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

                        if msg_type == "AgentProgressEvent":
                            # hiện ...
                            await sio.emit(
                                "running",
                                {
                                    "content": data,
                                    "role": MessageRole.ASSISTANT.value,
                                },
                                to=socket_id,
                            )
                        # elif msg_type == "AgentInput":
                        #     pass
                        elif msg_type == "AgentStream":
                            thinking_delta = data.get("thinking_delta", None)
                            if thinking_delta:
                                # also save
                                thinking_step = ThinkingStep(
                                    title="Thinking", description=thinking_delta
                                )
                                thinking_accum.append(thinking_step)
                                await sio.emit(
                                    "thinking",
                                    {
                                        "title": "Thinking",
                                        "description": thinking_delta,
                                    },
                                    to=socket_id,
                                )
                            else:  # reponse
                                response = data.get("response", "")
                                response_delta = data.get("delta", "")
                                # accumulate the delta
                                if response_delta:
                                    accum += response_delta
                                await sio.emit(
                                    "response",
                                    {
                                        "content": response,
                                        "content_delta": response_delta,
                                        "role": MessageRole.ASSISTANT.value,
                                    },
                                    to=socket_id,
                                )
                        elif msg_type == "ToolCall":
                            tool_id = data.get("tool_id", "")
                            description = data.get("description", "")
                            tool_step = ToolStep(
                                tool_name=tool_id, description=description, status="finished"
                            )
                            tools_accum.append(tool_step)

                            await sio.emit(
                                "tool_call",
                                {
                                    "tool_id": tool_id,
                                    "tool_name": tool_id,
                                },
                                to=socket_id,
                            )
                        elif msg_type == "ToolCallResult":
                            # show toolicon trước
                            # mốt show list hình ảnh từ s3, video + các timestamp

                            raw_output = data.get("tool_output", {}).get(
                                "raw_output", {}
                            )

                            if isinstance(raw_output, dict):
                                summary = raw_output.get("summary", {})
                            else:
                                summary = {}
                            # check if object
                            media_type = summary.get("result_type", "")

                            s3_base = "s3://"
                            http_base = "http://100.113.186.28:9000/"

                            def format_tool_result(media):
                                if media_type == "image_search":
                                    image_groups = defaultdict(list)
                                    image_results = []
                                    for item in media:
                                        image_groups[item["video_id"]].append(item)

                                    for video_id, images in image_groups.items():
                                        url_list = []
                                        for item in images:
                                            url_list.append(
                                                item["minio_path"].replace(
                                                    s3_base, http_base
                                                )
                                            )
                                        image_block = ImageBlock(
                                            video_id=video_id, url=url_list
                                        )
                                        ai_message_blocks.append(image_block)
                                        image_results.append(image_block)
                                    return {
                                        "media_type": "image",
                                        "results": image_results,
                                    }
                                elif media_type == "segment_caption_search":
                                    # video_url = [
                                    #     item["minio_path"].replace(s3_base, http_base)
                                    #     for item in media
                                    # ]
                                    # hard coded map for testing now
                                    path = {
                                        "692ad412086ada3a309334ff": "s3://videotests/K01_V001.mp4",
                                        "692ad412086ada3a30933500": "s3://videotests/K01_V002.mp4",
                                        "692ad412086ada3a30933501": "s3://videotests/K01_V003.mp4",
                                        "692ad412086ada3a30933502": "s3://videotests/K01_V004.mp4",
                                    }
                                    # group similar video segments into one block
                                    video_groups = defaultdict(list)
                                    video_results = []
                                    for item in media:
                                        video_groups[item["video_id"]].append(item)

                                    for video_id, segments in video_groups.items():
                                        segment_list = []
                                        for item in segments:
                                            segment = VideoSegment(
                                                start=item["frame_range"]["start"],
                                                end=item["frame_range"]["end"],
                                                caption=item["caption_preview"],
                                            )
                                            segment_list.append(segment)
                                        # sort segment_list based on start time
                                        segment_list.sort(key = lambda x: x.start)
                                        video_block = VideoBlock(
                                            video_id=video_id,
                                            url=path[video_id].replace(
                                                s3_base, http_base
                                            ),
                                            fps=segments[0].get("fps", 30),
                                            segments=segment_list,
                                        )
                                        ai_message_blocks.append(video_block)
                                        video_results.append(video_block)

                                    return {
                                        "media_type": "video",
                                        "results": video_results,
                                    }
                                else:
                                    return {"media_type": "unknown", "results": []}

                            media = summary.get("top_matches", [])
                            formatted_media = format_tool_result(media)
                            if formatted_media["media_type"] in [
                                "image",
                                "video",
                            ]:
                                # Convert Pydantic models to dicts for JSON serialization
                                emit_data = {
                                    "media_type": formatted_media["media_type"],
                                    "results": [
                                        r.model_dump()
                                        for r in formatted_media["results"]
                                    ],
                                }
                                await sio.emit("media", emit_data, to=socket_id)
                            else:
                                tool_id = data.get("tool_id", "unknown_tool")
                                tool_name = data.get("tool_id", "unknown_tool")
                                description = data.get("description", "")
                                # also update tool result in ai message blocks
                                await sio.emit(
                                    "tool_result",
                                    {
                                        "tool_id": tool_id,
                                        "tool_name": tool_name,
                                        "description": description,
                                    },
                                    to=socket_id,
                                )
                        elif msg_type == "AgentOutput":
                            # save to database
                            ai_message = SessionMessage(
                                session_id=session_id,
                                role=MessageRole.ASSISTANT,
                                blocks=ai_message_blocks,
                            )
                            await app_state.chat_service.add_message(
                                session_id, "assistant", ai_message
                            )

                            # Emit stream_end to notify the client
                            await sio.emit(
                                "stream_end",
                                {"session_id": session_id},
                                to=socket_id,
                            )
                        else:
                            pass

                        # Update prev_msg_type at end of loop
                        prev_msg_type = msg_type

                    except Exception as e:
                        print("⚠️ parse error:", e)
                        traceback.print_exc()  # This will show the exact line number

        except Exception as e:
            await sio.emit(
                "error", {"message": "agent unreachable: " + str(e)}, to=socket_id
            )

    except Exception as e:
        await sio.emit("error", {"message": str(e)}, to=socket_id)
