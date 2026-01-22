"""
This code will contain all the chat models
1. Containing message blocks
2. Chat models
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Type, TypeVar, Literal, Annotated, Union
from beanie import Document
from datetime import datetime
from abc import abstractmethod, ABC
from llama_index.core.base.llms.types import MessageRole


from app.core.config import settings
from beanie import PydanticObjectId

from utils.random_name import random_chat_name


# chat session
class ChatHistory(Document):  
    user_id: PydanticObjectId

    name: str = Field(default_factory=random_chat_name)

    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    class Settings:
        name = settings.CHAT_COLLECTION_NAME
        indexes = [
            [("last_updated", -1)]
        ]
