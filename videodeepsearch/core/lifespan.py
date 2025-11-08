from fastapi import FastAPI
from contextlib import asynccontextmanager
from sqlalchemy import text
from llama_index.llms.google_genai import GoogleGenAI 

from videodeepsearch.tools.clients import *

from .app_state import Appstate
from .config.client_config import (
    image_milvus_config, 
    segment_caption_milvus_config, 
    postgres_client_config, 
    minio_storage_client_config, 
    external_image_embedding_config, 
    external_text_embedding_config
)

from .config.llm_config import (
    llm_configs
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    app_state = Appstate()

    external_image_embedding_client = ImageEmbeddingClient(
        base_url=external_image_embedding_config.external_image_embedding_client_base_url
    )
    external_text_embedding_client = TextEmbeddingClient(
        base_url=external_text_embedding_config.external_text_embedding_client_base_url
    )

    image_external_embeddings = ImageEmbeddingSettings(
        model_name=external_image_embedding_config.external_image_embedding_client_model_name,
        device=external_image_embedding_config.external_image_embedding_client_device,
        batch_size=external_image_embedding_config.external_image_embedding_client_batch_size
    )

    text_external_embeddings = TextEmbeddingSettings(
        model_name=external_text_embedding_config.external_text_embedding_client_model_name,
        device=external_text_embedding_config.external_text_embedding_client_device,
        batch_size=external_text_embedding_config.external_text_embedding_client_batch_size
    )
    app_state.external_client = ExternalEncodeClient(
        img_text_client=external_image_embedding_client,
        img_text_settings=image_external_embeddings,
        txt_settings=text_external_embeddings,
        txt_client=external_text_embedding_client
    )
    app_state.image_milvus_client = ImageMilvusClient(
        uri=image_milvus_config.image_milvus_uri,
        collection_name=image_milvus_config.image_milvus_collection_name,
        visual_param=image_milvus_config.image_milvus_visual_param,
        caption_param=image_milvus_config.image_milvus_caption_param,
        sparse_param=image_milvus_config.image_milvus_sparse_param
    )
    app_state.segment_milvus_client = SegmentCaptionImageMilvusClient(
        uri=segment_caption_milvus_config.segment_milvus_caption_uri,
        collection_name=segment_caption_milvus_config.segment_milvus_collection_name,
        sparse_param=segment_caption_milvus_config.segment_milvus_sparse_param,
        dense_param=segment_caption_milvus_config.segment_milvus_dense_param
    )
    app_state.postgres_client = PostgresClient(
        database_url=postgres_client_config.postgres_client_database_url
    )
    app_state.minio_client = StorageClient(
        host=minio_storage_client_config.minio_storage_client_host,
        port=minio_storage_client_config.minio_storage_client_port,
        access_key=minio_storage_client_config.minio_storage_client_access_key,
        secret_key=minio_storage_client_config.minio_storage_client_secret_key,
        secure=minio_storage_client_config.minio_storage_client_secure
    )

    # Client connection establishment
    await app_state.image_milvus_client.connect()
    await app_state.segment_milvus_client.connect()
    await app_state.external_client.connect()

    async with app_state.postgres_client.get_session() as session:
        result = await session.execute(text("SELECT version();"))
        version = result.scalar_one()
        print(f"🗄️ PostgreSQL connection successful.\n   Version: {version}")

    # llm init
    for llm_config in llm_configs:
        agent_name = llm_config.agent_name
        app_state.llm_instance[agent_name] = GoogleGenAI(
            model=llm_config.model_name,
            generation_config=llm_config.generation_config
        )
    
    
    


    

    
    








    yield 

