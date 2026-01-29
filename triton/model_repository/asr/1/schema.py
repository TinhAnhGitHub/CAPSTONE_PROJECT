from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, Any
from enum import Enum
from datetime import datetime


class ASRConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_size: int = Field(default=64, ge=1)
    left_context_size: int = Field(default=128, ge=0)
    right_context_size: int = Field(default=128, ge=0)
    total_batch_duration: int = Field(default=1800, ge=1)
    sample_rate: int = Field(default=16000, ge=1000)
    num_extraction_workers: int = Field(default=2, ge=1, le=8)
    num_asr_workers: int = Field(default=1, ge=1, le=4)



class TimestampedToken(BaseModel):
    text: str
    start: str # HH:MM:SS.ms
    end: str  # HH:MM:SS.ms
    start_frame: str
    end_frame: str

    @field_validator("start", "end")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """
        Ensure time strings are convertible to datetime objects.
        Accepts format: HH:MM:SS:ms
        """
        try:
            datetime.strptime(v, "%H:%M:%S.%f")
        except Exception:
            raise ValueError(f"Invalid timestamp format: {v} (expected HH:MM:SS:ms)")
        return v

class ASRResult(BaseModel):
    tokens: list[TimestampedToken]
    processing_time_seconds: float
    audio_duration_seconds: float


class ASRInferenceRequest(BaseModel):
    video_minio_url: Optional[str] = Field(None, description="s3:// URL or external URL to the input video")
    metadata: dict[str, Any] = Field(default_factory=dict)
    config: Optional[ASRConfig] = Field(default=None)


class ASRInferenceResponse(BaseModel):
    video_minio_url: str
    metadata: dict[str, Any]
    result: ASRResult
    status: str = "success"



class ProcessingStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"




class VideoFile(BaseModel):
    video_path: str = Field(..., description="Path to the video file")


class BatchProcessRequest(BaseModel):
    video_files: list[VideoFile] = Field(...)
    config: Optional[ASRConfig] = Field(default_factory=ASRConfig)
    batch_id: Optional[str] = Field(None, description="Custom batch identifier")


class FileProcessingStatus(BaseModel):
    video_path: str
    status: ProcessingStatus
    error_message: Optional[str] = None
    result: Optional[ASRResult] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress_percentage: float = Field(ge=0, le=100)

class BatchStatus(BaseModel):
    batch_id: str
    status: ProcessingStatus
    total_files: int
    completed_files: int
    failed_files: int
    progress_percentage: float = Field(ge=0, le=100)
    files: list[FileProcessingStatus]
    started_at: str
    estimated_completion: Optional[str] = None


class BatchResponse(BaseModel):
    batch_id: str
    message: str
    total_files: int


class StreamingUpdate(BaseModel):
    batch_id: str
    file_path: str
    status: ProcessingStatus
    progress: float
    result: Optional[ASRResult] = None
    error: Optional[str] = None
    timestamp: str






