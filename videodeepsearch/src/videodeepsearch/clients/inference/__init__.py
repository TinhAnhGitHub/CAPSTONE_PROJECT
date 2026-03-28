"""Inference clients for embedding services."""

from videodeepsearch.clients.inference.client import MMBertClient, QwenVLEmbeddingClient, SpladeClient
from videodeepsearch.clients.inference.schema import MMBertConfig, QwenVLEmbeddingConfig, SpladeConfig

__all__ = [
    # Clients
    "QwenVLEmbeddingClient",
    "MMBertClient",
    "SpladeClient",
    # Configs
    "QwenVLEmbeddingConfig",
    "MMBertConfig",
    "SpladeConfig",
]