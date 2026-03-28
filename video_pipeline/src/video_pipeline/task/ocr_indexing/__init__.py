"""OCR indexing task module."""

from video_pipeline.task.ocr_indexing.main import (
    OCRIndexingTask,
    ocr_indexing_chunk_task,
)

__all__ = [
    "OCRIndexingTask",
    "ocr_indexing_chunk_task",
]