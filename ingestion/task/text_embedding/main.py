from __future__ import annotations
import time
import asyncio
import io
import json
from typing import AsyncIterator, BinaryIO, Literal, cast
from tqdm.asyncio import tqdm
from core.artifact.persist import ArtifactPersistentVisitor
from core.artifact.schema import (
    ImageCaptionArtifact,
    SegmentCaptionArtifact,
    TextCapSegmentEmbedArtifact,
    TextCaptionEmbeddingArtifact,
)
from core.clients.base import BaseMilvusClient, BaseServiceClient
from core.clients.text_embed_client import TextEmbeddingRequest, TextEmbeddingResponse
from core.pipeline.base_task import BaseTask
from prefect.logging import get_run_logger
from pydantic import BaseModel

from task.common.util import cleanup_temp_file, fetch_object_from_s3

class TextEmbeddingSettings(BaseModel):
    model_name: str
    device: Literal['cuda', 'cpu'] 
    batch_size: int



class TextImageCaptionEmbeddingTask(BaseTask[
    list[ImageCaptionArtifact], TextCaptionEmbeddingArtifact
]):
    def __init__(
        self,
        artifact_visitor: ArtifactPersistentVisitor,
        **kwargs,
    ):
        super().__init__(
            name=TextImageCaptionEmbeddingTask.__name__,
            visitor=artifact_visitor,
             **kwargs
        )

    async def preprocess(
        self,
        input_data: list[ImageCaptionArtifact]
    ) -> list[TextCaptionEmbeddingArtifact]:
        result = []
        for img_artifact in input_data:
            timestamp = img_artifact.time_stamp
            text_embed_art = TextCaptionEmbeddingArtifact(
                time_stamp=timestamp,
                related_frame_fps=img_artifact.related_video_fps,
                frame_index=img_artifact.frame_index,
                image_caption_minio_url=img_artifact.minio_url_path,
                user_bucket=img_artifact.user_bucket,
                caption_id=img_artifact.artifact_id,
                artifact_type=TextCaptionEmbeddingArtifact.__name__,
                related_video_id=img_artifact.related_video_id,
                image_minio_url=img_artifact.image_minio_url,
                image_id=img_artifact.image_id
            )
            result.append(text_embed_art)
        return result
    

    async def execute(
        self,
        input_data: list[TextCaptionEmbeddingArtifact],
        client: BaseServiceClient | None | BaseMilvusClient
    ) -> AsyncIterator[tuple[TextCaptionEmbeddingArtifact, BinaryIO | None]]:
        logger = get_run_logger()
        logger.info("Starting text image caption embedding for %d artifacts", len(input_data))
        assert client is not None, "The execution required client service"
        assert isinstance(client, BaseServiceClient)


        batch: list[TextCaptionEmbeddingArtifact] = []
        batches: list[list[TextCaptionEmbeddingArtifact]]  = []

        bs = cast(int, self.kwargs.get('batch_size'))
        for artifact in input_data:
            batch.append(artifact)
            if len(batch) == bs:
                batches.append(batch[:])
                batch.clear()
                
        if batch:
            batches.append(batch[:])
        
        logger.info(f"There is a total of {len(batches)} for the image caption embedding")

        for batch in tqdm(batches, desc="Processing text image embedding..."):
            print(f"Start gathering image caption for te")
            start_time = time.time()
            caption_dict_paths = await asyncio.gather(
                *[
                    fetch_object_from_s3(
                        artifact.image_caption_minio_url,
                        self.visitor.minio_client,
                        suffix='.json',
                    )
                    for artifact in batch
                ]
            )
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Image caption embedding duration takes: {duration}")

            caption_str: list[str] = []
            try:
                for item_path in caption_dict_paths:
                    with open(item_path, 'r', encoding='utf-8') as fp:
                        caption_str.append(json.load(fp)['caption'])
            finally:
                for item_path in caption_dict_paths:
                    cleanup_temp_file(item_path)

            request = TextEmbeddingRequest(
                texts=caption_str,
                metadata={}
            )
            response = await client.make_request(
                method='POST',
                endpoint=client.inference_endpoint,
                request_data=request
            )
            
            parsed = TextEmbeddingResponse.model_validate(response)
            embedding_captions = parsed.embeddings
            
            
            for artifact, embedding in tqdm(zip(batch, embedding_captions), desc="Persiting image caption embedding..."):
                buffer = io.BytesIO(json.dumps(embedding).encode("utf-8"))
                buffer.seek(0)
                logger.debug("Generated embedding for caption artifact  artifact.caption_id")
                yield artifact, buffer
    
    async def postprocess(self, output_data: tuple[TextCaptionEmbeddingArtifact, BinaryIO | None]):
        artifact, data = output_data
        if data is None:
            return artifact
        await artifact.accept_upload(self.visitor, data)
        return artifact
    



class TextCaptionSegmentEmbeddingTask(
    BaseTask[
        list[SegmentCaptionArtifact], TextCapSegmentEmbedArtifact
    ]
):
    def __init__(
        self,
        artifact_visitor: ArtifactPersistentVisitor,
        **kwargs,
    ):
        super().__init__(
            name=TextCaptionSegmentEmbeddingTask.__name__,
            visitor=artifact_visitor,
             **kwargs
        )

    async def preprocess(
        self, 
        input_data:list[SegmentCaptionArtifact]
    ) -> list[TextCapSegmentEmbedArtifact]:
        result = []
        for seg_artifact in input_data:
            text_embed_art = TextCapSegmentEmbedArtifact(
                related_video_fps=seg_artifact.related_video_fps,
                start_time=seg_artifact.start_timestamp,
                end_time=seg_artifact.end_timestamp,
                start_frame=seg_artifact.start_frame,
                end_frame=seg_artifact.end_frame,
                related_segment_caption_url=seg_artifact.minio_url_path,
                user_bucket=seg_artifact.user_bucket,
                segment_cap_id=seg_artifact.artifact_id,
                artifact_type=TextCapSegmentEmbedArtifact.__name__,
                related_video_id=seg_artifact.related_video_id
            )
            result.append(text_embed_art)
        return result

    async def execute(
        self,
        input_data: list[TextCapSegmentEmbedArtifact],
        client: BaseServiceClient | None | BaseMilvusClient
    ) -> AsyncIterator[tuple[TextCapSegmentEmbedArtifact, BinaryIO | None]]:
        logger = get_run_logger()
        logger.info("Starting text segment caption embedding for %d artifacts", len(input_data))
        assert client is not None, "The execution required client service"
        assert isinstance(client, BaseServiceClient)


        batch: list[TextCapSegmentEmbedArtifact] = []
        batches: list[list[TextCapSegmentEmbedArtifact]]  = []
        bs = cast(int, self.kwargs.get('batch_size'))
        for artifact in input_data:
            
            batch.append(artifact)
            if len(batch) == bs:
                batches.append(batch[:])
                batch.clear()
                
        if batch:
            batches.append(batch[:])

        for batch in batches:
            print(f"Start gathering segment captoin for te")
            start_time = time.time()
            caption_dict_paths = await asyncio.gather(
                *[
                    fetch_object_from_s3(
                        artifact.related_segment_caption_url,
                        self.visitor.minio_client,
                        suffix='.json',
                    )
                    for artifact in batch
                ]
            )
            end_time = time.time()
            duration = end_time - start_time
            print(f"Segment caption embedding duration takes: {duration}")

            caption_str: list[str] = []
            try:
                for item_path in caption_dict_paths:
                    with open(item_path, 'r', encoding='utf-8') as fp:
                        caption_str.append(json.load(fp)['caption'])
            finally:
                for item_path in caption_dict_paths:
                    cleanup_temp_file(item_path)

            request = TextEmbeddingRequest(
                texts=caption_str,
                metadata={}
            )
            logger.debug(
                "Submitting segment embedding request with %d captions", 
                len(caption_str)
            )
            response = await client.make_request(
                method='POST',
                endpoint=client.inference_endpoint,
                request_data=request
            )
            logger.debug("Received embedding response for batch of size %d", len(caption_str))
            parsed = TextEmbeddingResponse.model_validate(response)
            embedding_captions = parsed.embeddings

            for artifact, embedding in tqdm(zip(batch, embedding_captions), desc="Persiting segment caption embedding..."):
                buffer = io.BytesIO(json.dumps(embedding).encode("utf-8"))
                buffer.seek(0)
                logger.debug(
                    "Generated segment embedding for artifact %s", 
                    artifact.segment_cap_id,
                )
                yield artifact, buffer

    async def postprocess(self, output_data: tuple[TextCapSegmentEmbedArtifact, BinaryIO | None]):
        artifact, data = output_data
        if data is None:
            return artifact
        await artifact.accept_upload(self.visitor, data)
        return artifact
