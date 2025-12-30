from typing import Generic, TypeVar
from pydantic import BaseModel, Field, PrivateAttr
from uuid import uuid4


class DataHandle(BaseModel, Generic[T]):

    handle_id: str = Field(default_factory=lambda: str(uuid4()))
    tool_used: ToolCall | None = Field(None)
    summary: str = Field(
        default_factory=str, description="Human-readable summary for the agent"
    )

    related_video_ids: list[str]

    _raw_data: T | None = PrivateAttr(default=None)
