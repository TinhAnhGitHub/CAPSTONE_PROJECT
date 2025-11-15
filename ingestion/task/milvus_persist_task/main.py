from __future__ import annotations

import json
from typing import Any, AsyncIterator, cast

from core.artifact.persist import ArtifactPersistentVisitor
from core.artifact.schema import (
    ImageEmbeddingArtifact,
    TextCapSegmentEmbedArtifact,
    TextCaptionEmbeddingArtifact,
)
from core.clients.base import BaseMilvusClient, BaseServiceClient
from core.pipeline.base_task import BaseTask
from prefect.logging import get_run_logger
from pydantic import BaseModel
from task.common.util import cleanup_temp_file, fetch_object_from_s3, fetch_object_from_s3_bytes
from tqdm.asyncio import tqdm


class ImageEmbeddingMilvusTask(
    BaseTask[
        tuple[
            list[ImageEmbeddingArtifact],
            list[TextCaptionEmbeddingArtifact]
        ], tuple[ImageEmbeddingArtifact, TextCaptionEmbeddingArtifact]
    ]
):
    def __init__(
        self,
        artifact_visitor: ArtifactPersistentVisitor,
        **kwargs
    ):
        super().__init__(
            name=ImageEmbeddingMilvusTask.__name__,
            visitor=artifact_visitor,
             **kwargs
        )

    async def preprocess(
        self, 
        input_data: tuple[
            list[ImageEmbeddingArtifact],
            list[TextCaptionEmbeddingArtifact]
        ]
    ) -> list[tuple[ImageEmbeddingArtifact, TextCaptionEmbeddingArtifact]]:
        result = []
        image_embed_artifacts, text_caption_artifacts = input_data
        image_map = {
            img.image_id: img for img in image_embed_artifacts
        }
        for text_cap in text_caption_artifacts:
            img = image_map[text_cap.image_id]
            result.append(
                (img, text_cap)
            )
        return result

    async def execute(
        self, 
        input_data: list[tuple[ImageEmbeddingArtifact, TextCaptionEmbeddingArtifact]], 

        client: BaseMilvusClient | None | BaseServiceClient
    ) -> AsyncIterator[tuple[ImageEmbeddingArtifact, TextCaptionEmbeddingArtifact]]:
        
        assert client is not None, "Client required"
        assert isinstance(client, BaseMilvusClient), "Client must be from MilvusClient"
        await client.create_collection_if_not_exists()
        logger = get_run_logger()
        logger.info(
            "Starting Milvus insertion for %d image/text embedding pairs",
            len(input_data),
        )

        batch: list[dict[str, Any]] = []
        batch_artifacts: list[tuple[ImageEmbeddingArtifact, TextCaptionEmbeddingArtifact]] = []


        ingest_batch_size = cast(int, self.kwargs.get('ingest_batch_size'))
        

        for image_artifact, text_artifact in tqdm(input_data, desc='Preparing batch...'):
            image_embedding_bytes = await fetch_object_from_s3_bytes(image_artifact.minio_url_path, self.visitor.minio_client)
            text_caption_embedding_bytes = await fetch_object_from_s3_bytes(
                text_artifact.minio_url_path, self.visitor.minio_client
            )
            caption_dict_path = await fetch_object_from_s3(
                text_artifact.image_caption_minio_url,
                self.visitor.minio_client,
                suffix='.json'
            )
            try:
                with open(caption_dict_path, 'r', encoding='utf-8') as caption_file:
                    caption_data = json.load(caption_file)
            finally:
                cleanup_temp_file(caption_dict_path)
            
            caption_text = caption_data['caption']
            image_embedding = json.loads(image_embedding_bytes.decode('utf-8'))
            text_caption_embedding = json.loads(text_caption_embedding_bytes.decode('utf-8'))

            record = {
                'id': image_artifact.image_id,
                'related_video_id': image_artifact.related_video_id,
                'image_minio_url': text_artifact.image_minio_url,
                'user_bucket': image_artifact.user_bucket,
                'frame_index': image_artifact.frame_index,
                'timestamp': image_artifact.time_stamp,
                'visual_embedding_field': image_embedding,
                'caption_embedding_field': text_caption_embedding,
                'image_caption':  caption_text
            }

            batch.append(record)
            batch_artifacts.append((image_artifact, text_artifact))
            if len(batch) >= ingest_batch_size:
                
                ids = await client.insert_vectors(batch)
                logger.info(
                    "Inserted batch of %d vectors into %s",
                    len(ids),
                    client.collection_name,
                )
                for art in batch_artifacts:
                    yield art
                batch.clear()
                batch_artifacts.clear()

        if batch:
            
            logger.debug("Inserting remaining %d vectors", len(batch))
            ids = await client.insert_vectors(batch)
            logger.info(
                "Inserted final batch of %d vectors into %s",
                len(ids),
                client.collection_name,
            )
            for art in batch_artifacts:
                yield art


    async def postprocess(self, output_data: tuple[ImageEmbeddingArtifact, TextCaptionEmbeddingArtifact]) -> tuple[ImageEmbeddingArtifact, TextCaptionEmbeddingArtifact]:
        return output_data


class TextSegmentCaptionMilvusTask(
    BaseTask[
        list[TextCapSegmentEmbedArtifact], TextCapSegmentEmbedArtifact
    ]
):
    def __init__(
        self,
        artifact_visitor: ArtifactPersistentVisitor,
        **kwargs,
    ):
        super().__init__(
            name=TextSegmentCaptionMilvusTask.__name__,
            visitor=artifact_visitor,
             **kwargs
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
        logger = get_run_logger()
        logger.info(
            "Starting Milvus insertion for %d segment caption embeddings",
            len(input_data),
        )
        
        batch: list[dict[str, Any]] = []
        batch_artifacts: list[TextCapSegmentEmbedArtifact] = []
        ingest_batch_size = cast(int, self.kwargs.get('ingest_batch_size'))

        for artifact in input_data:
            exists = await client.exists(  # type: ignore[attr-defined]
                id_= artifact.artifact_id,
                related_video_id=artifact.related_video_id,
                user_bucket=artifact.user_bucket
            )

            if exists:
                logger.debug(
                    "Segment caption embedding %s already persisted; skipping",
                    artifact.artifact_id,
                )
                yield artifact
                continue
            
            embedding_bytes = await fetch_object_from_s3_bytes(artifact.minio_url_path, self.visitor.minio_client)
            embedding = json.loads(embedding_bytes.decode('utf-8'))
            
            caption_dict_path = await fetch_object_from_s3(
                artifact.related_segment_caption_url,
                self.visitor.minio_client,
                suffix='.json'
            )
            try:
                with open(caption_dict_path, 'r', encoding='utf-8') as caption_file:
                    caption_data = json.load(caption_file)
            finally:
                cleanup_temp_file(caption_dict_path)
            caption_text = caption_data.get('caption', '')


            record = {
                'id': artifact.artifact_id,
                "start_frame": artifact.start_frame,
                "end_frame": artifact.end_frame,
                "start_time": artifact.start_time,
                "end_time": artifact.end_time,
                'related_video_id': artifact.related_video_id,
                "segment_caption": caption_text,
                "segment_caption_minio_url": artifact.related_segment_caption_url,
                "caption_embedding_field": embedding,
                'user_bucket': artifact.user_bucket
            }

            batch.append(record)
            batch_artifacts.append(artifact)

            if len(batch) >= ingest_batch_size:
            
                ids = await client.insert_vectors(batch)
                logger.info(
                    "Inserted batch of %d segment caption vectors into %s",
                    len(ids),
                    client.collection_name,
                )
                for art in batch_artifacts:
                    yield art
              
                batch.clear()
                batch_artifacts.clear()

        if batch:
            ids = await client.insert_vectors(batch)
            logger.info(
                "Inserted final batch of %d segment caption vectors into %s",
                len(ids),
                client.collection_name,
            )
            for art in batch_artifacts:
                yield art
            

    async def postprocess(self, output_data: TextCapSegmentEmbedArtifact) -> TextCapSegmentEmbedArtifact:
        return output_data
