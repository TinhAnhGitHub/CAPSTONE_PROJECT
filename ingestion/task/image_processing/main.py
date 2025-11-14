import json
from io import BytesIO
from typing import AsyncIterator, cast
from tqdm import tqdm
import time
from core.artifact.persist import ArtifactPersistentVisitor
from core.artifact.schema import AutoshotArtifact, ImageArtifact
from core.clients.base import BaseMilvusClient, BaseServiceClient
from core.pipeline.base_task import BaseTask
from prefect.logging import get_run_logger

from .util import get_segment_frame_indices, extract_frames_async
from task.common.util import cleanup_temp_file, get_video_bytes, fetch_object_from_s3


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
             **kwargs
        )

    async def preprocess(self, input_data: list[AutoshotArtifact]) -> dict[str, list[ImageArtifact]]:
        logger = get_run_logger()
        result: dict[str, list[ImageArtifact]] = {}
        

        for shot_artifact in input_data:
            shot_art_url = shot_artifact.minio_url_path
            segments_path = await fetch_object_from_s3(shot_art_url, self.visitor.minio_client, suffix='.json')
            try:
                with open(segments_path, 'r', encoding='utf-8') as f:
                    segments = json.load(f)['segments']
            finally:
                cleanup_temp_file(segments_path)
            logger.debug("Loaded %d segments for autoshot artifact %s", len(segments), shot_artifact.artifact_id)
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

                    list_images.append(image_artifact)
                if shot_artifact.related_video_minio_url not in result:
                    result[shot_artifact.related_video_minio_url] = []
                result[shot_artifact.related_video_minio_url].extend(list_images)
                logger.debug(
                    "Prepared %d images for video %s (segment %d)",
                    len(list_images),
                    shot_artifact.related_video_minio_url,
                    i,
                )
        return result

    async def execute(self, input_data: dict[str, list[ImageArtifact]], client: BaseServiceClient | None| BaseMilvusClient ) -> AsyncIterator[tuple[ImageArtifact, bytes | None]]:
        logger = get_run_logger()
        logger.debug("Executing image extraction for %d videos", len(input_data))
        for video_minio_path, img_artifacts in input_data.items():
            if not img_artifacts:  
                continue
            
            not_process_images = []

            for artifact in tqdm(img_artifacts, desc=f"Image processing for video {video_minio_path}... "):
                exist =  await artifact.accept_check_exist(self.visitor)
                if exist:
                    yield artifact, None
                    continue
                not_process_images.append(artifact)

            if not not_process_images:
                continue

            local_video_bytes = await get_video_bytes(
                video_minio_path,
                self.visitor.minio_client,
            )
            
            start_time = time.time()
            indices = [artifact.frame_index for artifact in not_process_images]
            print(f"Len indices: {len(indices)}")
            frames = await extract_frames_async(local_video_bytes, indices)
            end_time = time.time()
            print(f"Duration of processing a single request: {end_time - start_time}")

            for artifact, frame_byte in zip(not_process_images, frames):
                yield artifact, frame_byte
            
            
    async def postprocess(self, output_data: tuple[ImageArtifact, bytes | None]) -> ImageArtifact:
        artifact, image = output_data
        if image is None:
            return artifact
        await artifact.accept_upload(visitor=self.visitor, upload_file=BytesIO(image))
        return artifact
    
