from __future__ import annotations
from typing import Any, TYPE_CHECKING, Callable, Coroutine





class Appstate:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    context_file_management: FileSystemContextStore = None #type:ignore

    external_client: ExternalEncodeClient = None  #type:ignore
    image_milvus_client: ImageMilvusClient = None #type:ignore
    segment_milvus_client: SegmentCaptionImageMilvusClient = None #type:ignore
    postgres_client: PostgresClient = None #type:ignore
    minio_client: StorageClient = None #type:ignore

    llm_instance: dict[str, FunctionCallingLLM] = {} #type:ignore


def get_app_state() -> Appstate:
    app_state = Appstate()
    return app_state

