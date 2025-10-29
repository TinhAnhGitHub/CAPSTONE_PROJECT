from pydantic import BaseModel, Field




class ImageMilvusResponse(BaseModel):
    id: str
    related_video_id: str
    frame_index: int
    timestamp: str
    image_minio_url: str
    user_bucket: str
    image_caption: str
    score: float



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



class SegmentCaptionMilvusResponse(BaseModel):
    id: str
    start_frame: int
    end_frame: int
    start_time: str
    end_time: str
    related_video_id: str
    segment_caption:str
    segment_caption_minio_url: str
    user_bucket:str
    score: float




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

