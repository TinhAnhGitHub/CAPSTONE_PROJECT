from __future__ import annotations
from typing import Any
from core.clients.progress_client import ProgressClient
from 

class AppState:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    storage_client: Any = None
    artifact_tracker: Any = None
    artifact_visitor: Any = None
    artifact_deleter: Any = None

    video_ingestion_task: Any = None
    autoshot_task: Any = None
    asr_task: Any = None
    image_processing_task: Any = None
    segment_caption_llm_task: Any = None
    image_caption_llm_task: Any = None

    image_embedding_task: Any = None
    text_image_caption_embedding_task: Any = None
    text_caption_segment_embedding_task: Any = None

    image_embedding_milvus_task: Any = None
    text_image_caption_milvus_task: Any = None
    text_segment_caption_milvus_task: Any = None

    image_milvus_client: Any = None
    seg_milvus_client: Any = None

    base_client_config: Any = None
    progress_client: ProgressClient = None #type:ignore

