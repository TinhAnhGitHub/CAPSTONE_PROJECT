"""Audio Transcript Embedding Task Module.

Provides task for embedding ASR transcript text using mmBERT dense embeddings.
"""

from .main import (
    AudioTranscriptEmbeddingTask,
    audio_transcript_embedding_chunk_task,
    AUDIO_TRANSCRIPT_EMBEDDING_CONFIG,
)

__all__ = [
    "AudioTranscriptEmbeddingTask",
    "audio_transcript_embedding_chunk_task",
    "AUDIO_TRANSCRIPT_EMBEDDING_CONFIG",
]