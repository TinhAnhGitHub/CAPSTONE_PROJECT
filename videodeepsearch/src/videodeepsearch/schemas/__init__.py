"""Pydantic schemas for videodeepsearch."""

from .artifacts import BaseInterface, ImageInterface, SegmentInterface, ImageBytes

__all__ = [
    "BaseInterface",
    "ImageInterface",
    "SegmentInterface",
    "ImageBytes",
]