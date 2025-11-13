from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Type, TypeVar, Literal, Annotated, Union
from beanie import Document, PydanticObjectId
from datetime import datetime
from abc import abstractmethod, ABC
from llama_index.core.base.llms.types import (
    MessageRole,
    ImageBlock,
    ContentBlock,
    TextBlock,
    VideoBlock,
)

from app.core.config import settings


# T = TypeVar("T", bound="ChatBlock")
# class ChatBlock(BaseModel, ABC):
#     """
#     BaseClass for any message block
#     """

#     block_type: Any

#     @abstractmethod
#     def __str__(self) -> str:
#         """String representation of the message block"""

#     @classmethod
#     @abstractmethod
#     def from_str_cls(cls: Type[T], raw: str) -> T:
#         """
#         Reconstruct object from a string
#         """


# class TextBlock(ChatBlock):
#     block_type: Literal["text"] = "text"
#     text_content: str = Field(
#         ..., description="The main content of the text message block"
#     )

#     def __str__(self) -> str:
#         return self.text_content

#     @classmethod
#     def from_str_cls(cls, raw: str) -> TextBlock:
#         return cls(text_content=raw)


# class ImageBlock(ChatBlock):
#     block_type: Literal["image"] = "image"
#     image_urls: list[str] = Field(..., description="List of image URLs")

#     def __str__(self) -> str:
#         return "\n".join(self.image_urls)

#     @classmethod
#     def from_str_cls(cls, raw: str) -> ImageBlock:
#         urls = [url.strip() for url in raw.split(",") if url.strip()]
#         return cls(image_urls=urls)


# class VideoBlock(ChatBlock):
#     block_type: Literal["video"] = "video"
#     video_urls: list[str] = Field(..., description="The URL of the video")

#     def __str__(self) -> str:
#         return "\n".join(self.video_urls)

#     @classmethod
#     def from_str_cls(cls, raw: str) -> "VideoBlock":
#         urls = [url.strip() for url in raw.split(",") if url.strip()]
#         return cls(video_urls=urls)


# CONTENT_BLOCK = Annotated[
#     Union[TextBlock, ImageBlock, VideoBlock], Field(discriminator="block_type")
# ]


class SessionMessage(Document):
    """
    This is the Message that contains many type of messages, in a round
    user -> AI: 1 session message
    AI -> user: 1 session message
    """
    # message_id: auto gen
    session_id: PydanticObjectId | None # chỉ vào chathistory # r sao ko de chat histỏy la 1 list o day

    role: MessageRole = Field(
        ..., description="Message role: Could be USER, SYSTEM, ASSISTANT, ..."
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    additional_kwargs: dict[str, Any] = Field(default_factory=dict)
    blocks: list[ContentBlock] = Field(default_factory=list)

    class Settings:
        name = settings.CHAT_MESSAGE_COLLECTION_NAME
        indexes = [
            "role",
            [("timestamp", -1)],
            [("last_updated", -1)]
        ]
