from __future__ import annotations
from typing import AsyncIterator
from pydantic import BaseModel
import os
from task.common.util import fetch_object_from_s3, parse_s3_url
from core.pipeline.base_task import BaseTask
from core.artifact.persist import  ArtifactPersistentVisitor
from core.artifact.schema import VideoArtifact
from core.clients.base import BaseServiceClient, BaseMilvusClient
from .util import get_video_fps
from prefect.logging import get_run_logger


class VideoInput(BaseModel):
    files: list[tuple[str,str]]
    user_id: str

def extract_extension(s3_link:str) ->str:
    return s3_link.split('.')[-1]
class VideoIngestionTask(BaseTask[VideoInput, VideoArtifact]):
    def __init__(
        self, 
        artifact_visitor: ArtifactPersistentVisitor,
        **kwargs,
    ):
        super().__init__(
            name=VideoIngestionTask.__name__,
            visitor=artifact_visitor,
             **kwargs
        )
    async def preprocess(
        self, input_data: VideoInput
    ) -> VideoInput:
        return input_data

    async def execute(
        self,
        input_data: VideoInput,
        client: BaseServiceClient | None | BaseMilvusClient
    ) -> AsyncIterator[tuple[VideoArtifact, dict, bool]]:
        user_bucket_name = input_data.user_id
        logger = get_run_logger()
        logger.info("Starting video ingestion execution for user %s with %d files", user_bucket_name, len(input_data.files))

        for video_info in input_data.files:

            video_id, video_s3_path = video_info
            logger.info("Processing video %s from %s", video_id, video_s3_path)
            _, object_name = parse_s3_url(video_s3_path)
            video_extension = extract_extension(video_s3_path)
            logger.debug("Detected extension %s for video %s", video_extension, video_id)
            video_tmp_path = await fetch_object_from_s3(
                s3_url=video_s3_path,
                storage=self.visitor.minio_client,
                suffix=f'.{video_extension}',
            )
            logger.debug("Downloaded video %s to temporary path %s", video_id, video_tmp_path)
            fps = get_video_fps(video_tmp_path)
            metadata = {
                'fps': fps,
                'extension': video_extension
            }
            os.remove(video_tmp_path)
            video_artifact = VideoArtifact(
                artifact_type=VideoArtifact.__name__,
                task_name=self.name,
                video_id=video_id,
                video_extension=video_extension,
                video_minio_url=video_s3_path,
                user_bucket=user_bucket_name,
                fps=fps,
                object_name=object_name
            )
            logger.debug("Prepared metadata for video %s: %s", video_id, metadata)
            exists = await video_artifact.accept_check_exist(self.visitor)
            logger.info("Video artifact %s exists=%s", video_artifact.artifact_id, exists)
            video_id = video_artifact.artifact_id
            yield video_artifact, metadata, exists

    async def postprocess(self, output_data: tuple[VideoArtifact, dict, bool]) -> VideoArtifact:
        artifact, metadata, data = output_data
        if data:
            return artifact
        await artifact.accept_upload(self.visitor, metadata)
        return artifact
