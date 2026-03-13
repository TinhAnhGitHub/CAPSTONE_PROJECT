"""
LLM Provider clients for the video pipeline.

This module provides unified LLM client interfaces using LlamaIndex:

- OpenRouterClient: Access to 100+ LLMs via OpenRouter API
- GeminiClient: Direct Google Gemini API access
- MoondreamClient: Vision-specific API for image captioning/detection

All clients follow a consistent async-first API pattern.
"""

from video_pipeline.core.client.llm_provider.openrouter import (
    OpenRouterClient,
    OpenRouterConfig,
    OpenRouterResult,
)
from video_pipeline.core.client.llm_provider.gemini import (
    GeminiClient,
    GeminiConfig,
    GeminiResult,
)
from video_pipeline.core.client.llm_provider.moondream import (
    MoondreamClient,
    MoondreamConfig,
    CaptionResult,
    DetectResult,
    BatchStatus,
    BoundingBox,
)

__all__ = [
    # OpenRouter
    "OpenRouterClient",
    "OpenRouterConfig",
    "OpenRouterResult",
    # Gemini
    "GeminiClient",
    "GeminiConfig",
    "GeminiResult",
    # Moondream
    "MoondreamClient",
    "MoondreamConfig",
    "CaptionResult",
    "DetectResult",
    "BatchStatus",
    "BoundingBox",
]