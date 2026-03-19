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


class SpladeConfig(BaseModel):
    """Configuration for SPLADE sparse embedding client (Triton)."""
    url: str = "localhost:8001"
    model_name: str = "splade"
    timeout: int = 30
    verbose: bool = False
    max_batch_size: int = 32


__all__ = [
    "QwenVLEmbeddingConfig",
    "MMBertConfig",
    "SpladeConfig",
]