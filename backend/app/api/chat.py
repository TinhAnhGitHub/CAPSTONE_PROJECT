from fastapi import APIRouter, HTTPException, Query
from fastapi.security import HTTPBearer
import jwt
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from llama_index.core.base.llms.types import MessageRole

from app.core.dependencies import ChatServiceDep, AgentDep
from app.model.chat_history import ChatHistory
from app.schema.chat import ChatRequest, ChatResponse, SessionInfo
from app.schema.user import ALGORITHM, SECRET_KEY
from app.model.session_message import SessionMessage
from llama_index.core.base.llms.types import MessageRole, TextBlock

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post("/send", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    chat_service: ChatServiceDep,
    agent: AgentDep
):
    try:
        user_message = SessionMessage(
            role=request.role,
            blocks=[TextBlock(text=request.message)]
        )

        await chat_service.add_message(request.session_id, user_message)

        ai_response = agent.chat(request.message)

        ai_message = SessionMessage(
            role=MessageRole.ASSISTANT, blocks=[TextBlock(text=ai_response)]
        )

        await chat_service.add_message(request.session_id, ai_message)

        return ChatResponse(
            session_id=request.session_id,
            response=ai_response,
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    chat_service: ChatServiceDep
):
    try:
        history = await chat_service.get_chat_history(session_id)
        if not history:
            raise HTTPException(status_code=404, detail="Session not found")
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=List[SessionInfo])
async def get_all_sessions(
    chat_service: ChatServiceDep,
    limit: int = Query(default=50, description="Maximum number of sessions to return")
):
    try:
        sessions = await chat_service.get_all_session_ids()
        return [
            SessionInfo(
                session_id=session.session_id,
                last_updated=session.last_updated,
                message_count=len(session.chat)
            )
            for session in sessions[:limit]
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    chat_service: ChatServiceDep
):
    try:
        deleted = await chat_service.delete_chat_history(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"message": "Session deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
