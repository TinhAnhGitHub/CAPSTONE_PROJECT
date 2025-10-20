from __future__ import annotations
from typing import Any, AsyncIterator, Callable
from tqdm.asyncio import tqdm
from loguru import logger
from pydantic import BaseModel, Field
import io
from  core.pipeline.base_task import BaseTask
from core.artifact.schema import (

    ImageEmbeddingArtifact, 
    TextCapSegmentEmbedArtifact, 
    TextCaptionEmbeddingArtifact,
)
from core.artifact.persist import ArtifactPersistentVisitor
from task.common.util import fetch_object_from_s3_bytes, fetch_object_from_s3
from core.clients.base import BaseMilvusClient, BaseServiceClient, MilvusCollectionConfig
import json
from core.config.logging import run_logger

class MilvusIndexSettings(BaseModel):
    host: str
    port: int
    user: str | None
    password:str | None
    db_name: str
    time_out: float
    ingest_batch_size: int



class ImageEmbeddingMilvusTask(
    BaseTask[
        list[ImageEmbeddingArtifact], ImageEmbeddingArtifact, MilvusIndexSettings
    ]
):
    def __init__(
        self,
        artifact_visitor: ArtifactPersistentVisitor,
        config_client: MilvusIndexSettings,
    ):
        super().__init__(
            name=ImageEmbeddingMilvusTask.__name__,
            visitor=artifact_visitor,
            config=config_client
        )
        
    

    async def preprocess(self, input_data: list[ImageEmbeddingArtifact]) -> list[ImageEmbeddingArtifact]:
        return input_data

    async def execute(self, input_data: list[ImageEmbeddingArtifact], client: BaseMilvusClient | None | BaseServiceClient) -> AsyncIterator[ImageEmbeddingArtifact]:
        
        assert client is not None, "Client required"
        assert isinstance(client, BaseMilvusClient), "Client must be from MilvusClient"
        await client.create_collection_if_not_exists()

        batch: list[dict[str, Any]] = []
        batch_artifacts: list[ImageEmbeddingArtifact] = []

        

        for artifact in tqdm(input_data, desc='Preparing batch...'):
            
            exists = await client.exists(  # type: ignore[attr-defined]
                id_= artifact.artifact_id,
                related_video_id=artifact.related_video_id,
                user_bucket=artifact.user_bucket
            )

            if exists:
                logger.debug(
                    f"Skipping duplicate: {artifact.related_video_name} (segment {artifact.segment_index})"
                )
                continue
            
            data_bytes = await fetch_object_from_s3_bytes(artifact.minio_url_path, self.visitor.minio_client)
            embedding = json.loads(data_bytes.decode('utf-8'))
            record = {
                "id": artifact.artifact_id,
                "related_video_name": artifact.related_video_name,
                "related_video_id": artifact.related_video_id,
                "segment_index": artifact.segment_index,
                "minio_url": artifact.minio_url_path,
                "embedding": embedding,
                'user_bucket': artifact.user_bucket
            }


            batch.append(record)
            batch_artifacts.append(artifact)

            if len(batch) >= self.config.ingest_batch_size:
                
                ids = await client.insert_vectors(batch)
                logger.info(
                    f"Inserted batch of {len(ids)} vectors into {client.config.collection_name}"
                )
                for art in batch_artifacts:
                    yield art
                batch.clear()
                batch_artifacts.clear()

        if batch:
            try:
                print("Inserting vectors")
                ids = await client.insert_vectors(batch)
                logger.info(
                    f"Inserted final batch of {len(ids)} vectors into {client.config.collection_name}"
                )
                for art in batch_artifacts:
                    yield art
            except Exception as e:
                logger.exception("Final batch insert failed", error=str(e))

    async def postprocess(self, output_data: ImageEmbeddingArtifact) -> ImageEmbeddingArtifact:
        return output_data
    

class TextImageCaptionMilvusTask(
    BaseTask[
        list[TextCaptionEmbeddingArtifact], TextCaptionEmbeddingArtifact, MilvusIndexSettings
    ]
):
    def __init__(
        self,
        artifact_visitor: ArtifactPersistentVisitor,
        config_client: MilvusIndexSettings,
    ):
        super().__init__(
            name=TextImageCaptionMilvusTask.__name__,
            visitor=artifact_visitor,
            config=config_client
        )

    async def preprocess(self, input_data: list[TextCaptionEmbeddingArtifact]) -> list[TextCaptionEmbeddingArtifact]:
        return input_data

    async def execute(
        self, 
        input_data: list[TextCaptionEmbeddingArtifact], 
        client: BaseMilvusClient | None | BaseServiceClient
    ) -> AsyncIterator[TextCaptionEmbeddingArtifact]:
        
        assert client is not None, "Client required"
        assert isinstance(client, BaseMilvusClient), "Client must be from MilvusClient"
        await client.create_collection_if_not_exists()
        
        batch: list[dict[str, Any]] = []
        batch_artifacts: list[TextCaptionEmbeddingArtifact] = []

        for artifact in input_data:
            exists = await client.exists(  # type: ignore[attr-defined]
                id_= artifact.artifact_id,
                related_video_id=artifact.related_video_id,
                user_bucket=artifact.user_bucket
            )

            if exists:
                logger.debug(
                    f"Skipping duplicate text caption: {artifact.related_video_name} (frame {artifact.frame_index})"
                )
                continue
            
            embedding_bytes = await fetch_object_from_s3_bytes(artifact.minio_url_path, self.visitor.minio_client)
            embedding = json.loads(embedding_bytes.decode('utf-8'))
            
            caption_dict_path = await fetch_object_from_s3(
                artifact.image_caption_minio_url, 
                self.visitor.minio_client, 
                suffix='.json'
            )
            caption_data = json.load(open(caption_dict_path, 'r', encoding='utf-8'))
            caption_text = caption_data.get('caption', '')

            record = {
                'id': artifact.artifact_id,
                "frame_index": artifact.frame_index,
                "related_video_name": artifact.related_video_name,
                'related_video_id': artifact.related_video_id,
                "caption": caption_text,
                "caption_minio_url": artifact.image_caption_minio_url,
                "embedding": embedding,
                'user_bucket': artifact.user_bucket
            }

            batch.append(record)
            batch_artifacts.append(artifact)

            if len(batch) >= self.config.ingest_batch_size:
            
                ids = await client.insert_vectors(batch)
                logger.info(
                    f"Inserted batch of {len(ids)} text caption vectors into {client.config.collection_name}"
                )
                for art in batch_artifacts:
                    yield art
              
                batch.clear()
                batch_artifacts.clear()

        if batch:
            try:
                ids = await client.insert_vectors(batch)
                logger.info(
                    f"Inserted final batch of {len(ids)} text caption vectors into {client.config.collection_name}"
                )
                for art in batch_artifacts:
                    yield art
            except Exception as e:
                logger.exception("Final batch insert failed", error=str(e))

    async def postprocess(self, output_data: TextCaptionEmbeddingArtifact) -> TextCaptionEmbeddingArtifact:
        return output_data


class TextSegmentCaptionMilvusTask(
    BaseTask[
        list[TextCapSegmentEmbedArtifact], TextCapSegmentEmbedArtifact, MilvusIndexSettings
    ]
):
    def __init__(
        self,
        artifact_visitor: ArtifactPersistentVisitor,
        config_client: MilvusIndexSettings,
    ):
        super().__init__(
            name=TextSegmentCaptionMilvusTask.__name__,
            visitor=artifact_visitor,
            config=config_client
        )

    async def preprocess(self, input_data: list[TextCapSegmentEmbedArtifact]) -> list[TextCapSegmentEmbedArtifact]:
        return input_data

    async def execute(
        self, 
        input_data: list[TextCapSegmentEmbedArtifact], 
        client: BaseMilvusClient | None | BaseServiceClient
    ) -> AsyncIterator[TextCapSegmentEmbedArtifact]:
        
        assert client is not None, "Client required"
        assert isinstance(client, BaseMilvusClient), "Client must be from MilvusClient"
        await client.create_collection_if_not_exists()
        
        batch: list[dict[str, Any]] = []
        batch_artifacts: list[TextCapSegmentEmbedArtifact] = []

        for artifact in input_data:
            exists = await client.exists(  # type: ignore[attr-defined]
                id_= artifact.artifact_id,
                related_video_id=artifact.related_video_id,
                user_bucket=artifact.user_bucket
            )

            if exists:
                logger.debug(
                    f"Skipping duplicate segment caption: {artifact.related_video_name} "
                    f"(frames {artifact.start_frame}-{artifact.end_frame})"
                )
                continue
            
            embedding_bytes = await fetch_object_from_s3_bytes(artifact.minio_url_path, self.visitor.minio_client)
            embedding = json.loads(embedding_bytes.decode('utf-8'))


            
            caption_dict_path = await fetch_object_from_s3(
                artifact.related_segment_caption_url, 
                self.visitor.minio_client, 
                suffix='.json'
            )
            caption_data = json.load(open(caption_dict_path, 'r', encoding='utf-8'))
            caption_text = caption_data.get('caption', '')


            record = {
                'id': artifact.artifact_id,
                "start_frame": artifact.start_frame,
                "end_frame": artifact.end_frame,
                "related_video_name": artifact.related_video_name,
                'related_video_id': artifact.related_video_id,
                "caption": caption_text,
                "segment_caption_minio_url": artifact.related_segment_caption_url,
                "embedding": embedding,
                'user_bucket': artifact.user_bucket
            }

            batch.append(record)
            batch_artifacts.append(artifact)

            if len(batch) >= self.config.ingest_batch_size:
            
                ids = await client.insert_vectors(batch)
                logger.info(
                    f"Inserted batch of {len(ids)} segment caption vectors into {client.config.collection_name}"
                )
                for art in batch_artifacts:
                    yield art
              
                batch.clear()
                batch_artifacts.clear()

        if batch:
            try:
                ids = await client.insert_vectors(batch)
                logger.info(
                    f"Inserted final batch of {len(ids)} segment caption vectors into {client.config.collection_name}"
                )
                for art in batch_artifacts:
                    yield art
            except Exception as e:
                logger.exception("Final batch insert failed", error=str(e))

    async def postprocess(self, output_data: TextCapSegmentEmbedArtifact) -> TextCapSegmentEmbedArtifact:
        return output_data