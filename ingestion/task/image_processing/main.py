import json
from typing import AsyncIterator, cast
from core.pipeline.base_task import BaseTask
from core.clients.base import BaseServiceClient, BaseMilvusClient
from core.artifact.persist import ArtifactPersistentVisitor 
from core.artifact.schema import AutoshotArtifact, ImageArtifact
from io import BytesIO
from pydantic import BaseModel
import asyncio
from .util import get_segment_frame_indices, read_frame
from task.common.util import fetch_object_from_s3 
from core.config.logging import run_logger


def frame_to_timecode(frame_index: int, fps: float) -> str:
    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")
    total_seconds = frame_index / fps

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


class ImageProcessingTask(BaseTask[list[AutoshotArtifact], ImageArtifact]):
    def __init__(
        self,
        artifact_visitor: ArtifactPersistentVisitor,
        **kwargs,
    ):
        super().__init__(
            name=ImageProcessingTask.__name__,
            visitor=artifact_visitor,
            kwargs=kwargs
        )

    async def preprocess(self, input_data: list[AutoshotArtifact]) -> dict[str, list[ImageArtifact]]:

        result: dict[str, list[ImageArtifact]] = {}
        

        for shot_artifact in input_data:
            shot_art_url = shot_artifact.minio_url_path
            segments_path = await fetch_object_from_s3(shot_art_url, self.visitor.minio_client, suffix='.json')
            with open(segments_path, 'r', encoding='utf-8') as f:
                segments = json.load(f)['segments']
            run_logger.debug(f'{segments=}')
            for i, (start,end) in enumerate(segments):
                
                num_img_per_segment = cast(int, self.kwargs.get('num_img_per_segment'))
                indices = get_segment_frame_indices(start=start,end=end,n=num_img_per_segment)

                list_images = []
                for idx in indices:

                    time_stamp = frame_to_timecode(frame_index=idx, fps=shot_artifact.related_video_fps)
                    image_artifact = ImageArtifact(
                        frame_index=idx,
                        extension='.webp',
                        related_video_id=shot_artifact.related_video_id,
                        related_video_minio_url=shot_artifact.related_video_minio_url,
                        related_video_extension=shot_artifact.related_video_extension,
                        autoshot_artifact_id=shot_artifact.artifact_id,
                        user_bucket=shot_artifact.user_bucket,
                        metadata={},
                        content_type="image/webp",
                        artifact_type=ImageArtifact.__name__,
                        timestamp=time_stamp,
                        related_video_fps=shot_artifact.related_video_fps,
                    ) 
                    # exist =  await image_artifact.accept_check_exist(self.visitor)
                    # if exist:
                    #     continue

                    list_images.append(image_artifact)
                if shot_artifact.related_video_minio_url not in result:
                    result[shot_artifact.related_video_minio_url] = []
                result[shot_artifact.related_video_minio_url].extend(list_images)
                run_logger.debug(f"Video: {shot_artifact.related_video_minio_url}, Images so far: {len(result[shot_artifact.related_video_minio_url])}")
        return result

    async def execute(self, input_data: dict[str, list[ImageArtifact]], client: BaseServiceClient | None| BaseMilvusClient ) -> AsyncIterator[tuple[ImageArtifact, bytes | None]]:
        run_logger.debug(f"{input_data}")
        for video_minio_path, img_artifacts in input_data.items():
            if not img_artifacts:  
                continue
            
            not_process_images = []

            for artifact in img_artifacts:
                exist =  await artifact.accept_check_exist(self.visitor)
                if exist:
                    yield artifact, None
                    continue
                else:
                    not_process_images.append(artifact)
            
            local_video = await fetch_object_from_s3(video_minio_path, self.visitor.minio_client, suffix=img_artifacts[0].related_video_extension) # group image comes from 1 video -> same video extension
            tasks = [read_frame(local_video, artifact.frame_index) for artifact in not_process_images ]
            frames = await asyncio.gather(*tasks)
            for artifact, frame_byte in zip(img_artifacts, frames):
                yield artifact, frame_byte
            
    async def postprocess(self, output_data: tuple[ImageArtifact, bytes | None]) -> ImageArtifact:
        artifact, image = output_data
        if image is None:
            return artifact
        await artifact.accept_upload(visitor=self.visitor, upload_file=BytesIO(image))
        return artifact
    
