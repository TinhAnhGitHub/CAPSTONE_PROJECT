from __future__ import annotations
from typing import Any, TYPE_CHECKING, Callable, Coroutine
from llama_index.core.llms.function_calling import FunctionCallingLLM

from videodeepsearch.tools.clients import * 
from videodeepsearch.agent.context.management import FileSystemContextStore





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


def get_external_client() -> ExternalEncodeClient:
    return get_app_state().external_client


def get_visual_image_client() -> ImageMilvusClient:
    return get_app_state().image_milvus_client


def get_segment_milvus() -> SegmentCaptionImageMilvusClient:
    return get_app_state().segment_milvus_client


def get_postgres_client() -> PostgresClient:
    return get_app_state().postgres_client


def get_storage_client() -> StorageClient:
    return get_app_state().minio_client

def get_llm_instance(name: str | None = None) -> FunctionCallingLLM:
    if name is None:
        return get_app_state().llm_instance['FINAL_RESPONSE_AGENT']
    llm = get_app_state().llm_instance.get(name)
    if llm is None:
        raise ValueError(f"Agent name: {name} does not have llm instance yet. Register them in the settings")
    return llm

def get_context_file_sys() -> FileSystemContextStore:
    return get_app_state().context_file_management