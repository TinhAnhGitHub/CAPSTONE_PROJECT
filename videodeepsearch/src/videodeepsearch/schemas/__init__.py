"""Pydantic schemas for videodeepsearch."""

from .artifacts import BaseInterface, ImageInterface, SegmentInterface, AudioInterface, ImageBytes

__all__ = [
    "BaseInterface",
    "ImageInterface",
    "SegmentInterface",
    "AudioInterface",
    "ImageBytes",
]