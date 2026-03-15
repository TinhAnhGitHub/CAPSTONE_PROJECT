"""Image OCR task module."""

from video_pipeline.task.image_ocr.main import (
    ImageOCRTask,
    image_ocr_chunk_task,
)

__all__ = [
    "ImageOCRTask",
    "image_ocr_chunk_task",
]