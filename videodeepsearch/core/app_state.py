from typing import Any 
from llama_index.core.llms.function_calling import FunctionCallingLLM

from videodeepsearch.tools.clients import * 
from videodeepsearch.tools.type.factory import ToolFactory

class Appstate:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    tool_factory: ToolFactory = None #type:ignore

    external_client: ExternalEncodeClient = None  #type:ignore
    image_milvus_client: ImageMilvusClient = None #type:ignore
    segment_milvus_client: SegmentCaptionImageMilvusClient = None #type:ignore
    postgres_client: PostgresClient = None #type:ignore
    minio_client: StorageClient = None #type:ignore

    # llm style
    llm_instance: dict[str, FunctionCallingLLM] = None


    
def get_app_state() -> Appstate:
    app_state = Appstate()
    return app_state