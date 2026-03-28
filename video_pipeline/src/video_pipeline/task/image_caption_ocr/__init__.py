"""Combined Image Caption + OCR Task."""

from .main import (
    ImageCaptionOCRTask,
    image_caption_ocr_chunk_task,
    IMAGE_CAPTION_OCR_CONFIG,
)

__all__ = [
    "ImageCaptionOCRTask",
    "image_caption_ocr_chunk_task",
    "IMAGE_CAPTION_OCR_CONFIG",
]