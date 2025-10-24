from fastapi import Depends
from typing import Annotated

from app.service.agent import Agent
from app.service.chat import ChatService
from app.service.user import UserService
from app.service.minio import Minio as MinioService

from app.core.lifespan import app_state
# from app.core.config import settings

def get_agent() -> Agent:
    if app_state.agent is None:
        raise RuntimeError("Agent not initialized")
    return app_state.agent

def get_chat_service() -> ChatService:
    if app_state.chat_service is None:
        raise RuntimeError("Chat service not initialized")
    return app_state.chat_service

def get_user_service() -> UserService:
    if app_state.user_service is None:
        raise RuntimeError("user service not initialized")
    return app_state.user_service

def get_minio_service() -> MinioService:
    if app_state.minio_service is None:
        raise RuntimeError("MinIO service not initialized")
    return app_state.minio_service

AgentDep = Annotated[Agent, Depends(get_agent)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
UserServiceDep = Annotated[UserService, Depends(get_user_service)]
