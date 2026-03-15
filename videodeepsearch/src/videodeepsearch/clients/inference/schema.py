"""Inference client schemas for embedding configurations."""
from __future__ import annotations
from pydantic import BaseModel

class QwenVLEmbeddingConfig(BaseModel):
    """Configuration for QwenVL embedding client."""
    base_url: str


class MMBertConfig(BaseModel):
    """Configuration for MMBert text embedding client."""
    model_name: str = "mmbert"
    base_url: str = "http://localhost:8100"


__all__ = [
    "QwenVLEmbeddingConfig",
    "MMBertConfig",
]