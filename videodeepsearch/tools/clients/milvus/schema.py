from __future__ import annotations
from pydantic import BaseModel




class ImageFilterCondition(BaseModel):
    related_video_id: list[str] | None = None
    user_bucket: str | None = None

    def to_expr(self) -> str:
        parts = []
        if self.related_video_id:
            ids = ', '.join(f'"{v}"' for v in self.related_video_id)
            parts.append(f"related_video_id in [{ids}]")
        if self.user_bucket:
            parts.append(f'user_bucket == "{self.user_bucket}"')
        return " and ".join(parts) if parts else ""


class SegmentCaptionFilterCondition(BaseModel):
    related_video_id: list[str] | None = None
    user_bucket: str | None = None

    def to_expr(self) -> str:
        parts = []
        if self.related_video_id:
            ids = ', '.join(f'"{v}"' for v in self.related_video_id)
            parts.append(f"related_video_id in [{ids}]")
        if self.user_bucket:
            parts.append(f'user_bucket == "{self.user_bucket}"')
        return " and ".join(parts) if parts else ""

