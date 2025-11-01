from __future__ import annotations
from typing import AsyncIterator, BinaryIO, cast, Callable
from pathlib import Path
from pydantic import BaseModel
from fastapi import UploadFile
import os
from task.common.util import fetch_object_from_s3
from prefect import task
from loguru import logger
from core.pipeline.base_task import BaseTask
from core.artifact.persist import  ArtifactPersistentVisitor
from core.artifact.schema import VideoArtifact
from core.clients.base import BaseServiceClient, BaseMilvusClient
from .config import VideoIngestionSettings
from .util import get_video_fps
from core.management.progress import ProcessingStage
from core.app_state import AppState

# tracker = AppState().progress_tracker

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
            kwargs=kwargs
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

        for video_info in input_data.files:

            video_id, video_s3_path = video_info
            video_extension = extract_extension(video_s3_path)
            
            video_tmp_path = await fetch_object_from_s3(
                s3_url=video_s3_path,
                storage=self.visitor.minio_client,
                suffix=f'.{video_extension}'
            )
            fps = get_video_fps(video_tmp_path)
            metadata = {
                'fps': fps,
                'extension': video_extension
            }
            os.remove(video_tmp_path)

            try:
                video_artifact = VideoArtifact(
                    artifact_type=VideoArtifact.__name__,
                    task_name=self.name,
                    video_id=video_id,
                    video_extension=video_extension,
                    video_minio_url=video_s3_path,
                    user_bucket=user_bucket_name,
                    fps=fps
                )
                
                exists = await video_artifact.accept_check_exist(self.visitor)
                print(f"Exists: {exists}")
                video_id = video_artifact.artifact_id
                
                yield video_artifact, metadata, exists
            except Exception as e:
                logger.exception(f" Failed to process video {video_s3_path}: {e}")
                continue

    async def postprocess(self, output_data: tuple[VideoArtifact, dict, bool]) -> VideoArtifact:
        artifact, metadata, data = output_data
        if data:
            return artifact
        await artifact.accept_upload(self.visitor,metadata)
        return artifact
