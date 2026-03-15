"""Inference clients for embedding services."""

from videodeepsearch.clients.inference.client import MMBertClient, QwenVLEmbeddingClient
from videodeepsearch.clients.inference.schema import MMBertConfig, QwenVLEmbeddingConfig

__all__ = [
    # Clients
    "QwenVLEmbeddingClient",
    "MMBertClient",
    # Configs
    "QwenVLEmbeddingConfig",
    "MMBertConfig",
]