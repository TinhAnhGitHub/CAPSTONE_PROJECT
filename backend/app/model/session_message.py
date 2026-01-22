from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Optional, Type, TypeVar, Literal, Annotated, Union
from beanie import Document, PydanticObjectId
from datetime import datetime
from abc import abstractmethod, ABC
from llama_index.core.base.llms.types import (
    MessageRole,
    # ContentBlock,
    TextBlock,
    # ImageBlock,
    AudioBlock,
    DocumentBlock,
    CachePoint,
    CitableBlock,
    CitationBlock,
)
from llama_index.core.bridge.pydantic import (
    AnyUrl,
    Field,
    FilePath,
)

from llama_index.core.tools import ToolOutput

from app.core.config import settings


class ToolCallBlock(BaseModel):
    """All tool calls are surfaced."""

    block_type: Literal["tool_call"] = "tool_call"
    tool_name: str
    tool_kwargs: dict
    tool_call_id: str


class VideoSegment(BaseModel):
    start: float
    end: float
    caption: Optional[str] = None


class ThinkingStep(BaseModel):
    title: str
    description: str | None = None


class ThinkingBlock(BaseModel):
    """A representation of thinking state as a sequence of steps."""
    block_type: Literal["thinking"] = "thinking"
    steps: list[ThinkingStep] = []


class ToolStep(BaseModel):
    tool_name: str
    description: str | None = None


class ToolsBlock(BaseModel):
    """A representation of tools state as a sequence of tool usages."""

    block_type: Literal["tools"] = "tools"
    tools: list[ToolStep] = []


class VideoBlock(BaseModel):
    """A representation of video data to directly pass to/from the LLM."""

    block_type: Literal["video"] = "video"
    # video: bytes | None = None
    # path: FilePath | None = None
    video_id: str | None = None
    url: AnyUrl | str | None = None
    video_mimetype: str | None = None
    # detail: str | None = None
    fps: int | None = None
    segments: list[VideoSegment] | None = None


class ImageBlock(BaseModel):
    """A representation of image data to directly pass to/from the LLM."""

    block_type: Literal["image"] = "image"
    video_id: str | None = None
    # image: bytes | None = None
    # path: FilePath | None = None
    url: list[AnyUrl | str] | None = None
    # image_mimetype: str | None = None
    # detail: str | None = None


class ToolCallResultBlock(BaseModel):
    """Tool call result."""

    block_type: Literal["tool_call_result"] = "tool_call_result"
    tool_name: str
    tool_kwargs: dict
    tool_id: str
    tool_output: ToolOutput
    return_direct: bool


# Extended content block that includes custom types
ContentBlock = Annotated[
    Union[
        TextBlock,
        # ImageBlock,
        AudioBlock,
        DocumentBlock,
        CachePoint,
        CitableBlock,
        CitationBlock,
        ImageBlock,
        VideoBlock,
        ToolCallBlock,
        ToolCallResultBlock,
        ThinkingBlock,
        ToolsBlock,
    ],
    Field(discriminator="block_type"),
]


class SessionMessage(Document):
    """
    This is the Message that contains many type of messages, in a round
    user -> AI: 1 session message
    AI -> user: 1 session message
    """

    # message_id: auto gen
    session_id: (
        PydanticObjectId | None
    )  # chỉ vào chathistory # r sao ko de chat histỏy la 1 list o day

    role: MessageRole = Field(
        ..., description="Message role: Could be USER, SYSTEM, ASSISTANT, ..."
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    additional_kwargs: dict[str, Any] = Field(default_factory=dict)
    blocks: list[ContentBlock] = Field(default_factory=list)

    class Settings:
        name = settings.CHAT_MESSAGE_COLLECTION_NAME
        indexes = ["role", [("timestamp", -1)], [("last_updated", -1)]]
