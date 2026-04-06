from pydantic import BaseModel
from llama_index.core.base.llms.types import MessageRole
from datetime import datetime



class ChatRequest(BaseModel):
    session_id: str
    message: str
    role: MessageRole = MessageRole.USER


class ChatResponse(BaseModel):
    session_id: str
    response: str
    timestamp: datetime


class SessionInfo(BaseModel):
    session_id: str
    last_updated: datetime
    message_count: int