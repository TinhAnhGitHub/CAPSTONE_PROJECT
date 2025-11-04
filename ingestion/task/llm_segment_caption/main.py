from __future__ import annotations

import json
from typing import AsyncIterator, Literal, cast

from core.artifact.persist import ArtifactPersistentVisitor
from core.artifact.schema import ASRArtifact, AutoshotArtifact, SegmentCaptionArtifact
from core.clients.base import BaseMilvusClient, BaseServiceClient
from core.pipeline.base_task import BaseTask
from prefect.logging import get_run_logger
from prefect_agent.service_llm.schema import LLMRequest, LLMResponse, LLMSingleRequest
from pydantic import BaseModel
from tqdm import tqdm

from task.common.util import cleanup_temp_file, fetch_object_from_s3

from .prompt import SEGMENT_CAPTION_PROMPT
from .util import extract_images, return_related_asr_with_shot


class LLMCaptionSettings(BaseModel):
    model_name: str
    device: Literal['cuda', 'cpu']
    image_per_segments: int

class ShotASRInput(BaseModel):
    list_asrs: list[ASRArtifact]
    lists_autoshots: list[AutoshotArtifact]



def frame_to_timecode(frame_index: int, fps: float) -> str:
    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")
    total_seconds = frame_index / fps

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


class SegmentCaptionLLMTask(BaseTask[
    ShotASRInput, SegmentCaptionArtifact,
]):
    def __init__(
        self,
        artifact_visitor: ArtifactPersistentVisitor,
        **kwargs,
    ):
        super().__init__(
            name=SegmentCaptionLLMTask.__name__,
            visitor=artifact_visitor,
            **kwargs,
        )
    
    async def preprocess(self, input_data: ShotASRInput) -> list[SegmentCaptionArtifact]:
        logger = get_run_logger()
        list_asr, list_autoshot = input_data.list_asrs, input_data.lists_autoshots        
        assert len(list_asr) == len(list_autoshot), "Not equal error"
       

        result = []
        for asr, shot in zip(list_asr, list_autoshot):
            assert asr.related_video_id == shot.related_video_id, "Each pair should have the same related video id"
            

            asr_dict_path = await fetch_object_from_s3(asr.minio_url_path, self.visitor.minio_client, suffix='.json')
            autoshot_dict_path = await fetch_object_from_s3(shot.minio_url_path, self.visitor.minio_client, suffix='.json')

            try:
                with open(asr_dict_path, 'r', encoding='utf-8') as fp:
                    asr_dict = json.load(fp)
                with open(autoshot_dict_path, 'r', encoding='utf-8') as fp:
                    autoshot_dict = json.load(fp)
            finally:
                cleanup_temp_file(asr_dict_path)
                cleanup_temp_file(autoshot_dict_path)


            tokens = asr_dict['tokens']
            segments: list[tuple[int,int]] = autoshot_dict['segments']

            for start_frame, end_frame in segments: 
                related_asr = return_related_asr_with_shot(
                    asr_tokens=tokens, #type:ignore
                    start_frame=start_frame,
                    end_frame=end_frame
                )

                start_time_stamp = frame_to_timecode(frame_index=start_frame, fps=shot.related_video_fps)
                end_time_stamp = frame_to_timecode(frame_index=end_frame, fps=shot.related_video_fps)


                artifact = SegmentCaptionArtifact(
                    
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_timestamp=start_time_stamp,
                    end_timestamp=end_time_stamp,
                    related_video_fps=shot.related_video_fps,
                    related_asr=related_asr,
                    related_video_minio_url=shot.related_video_minio_url,
                    user_bucket=asr.user_bucket,
                    related_video_extension=shot.related_video_extension,
                    autoshot_artifact_id=shot.artifact_id,
                    asr_artifact_id=asr.artifact_id,
                    artifact_type=SegmentCaptionArtifact.__name__,
                    related_video_id=shot.related_video_id
                )
                result.append(artifact)

        logger.info("Prepared %d segment caption artifacts", len(result))
        return result

    async def _process_single_artifact(self, artifact: SegmentCaptionArtifact) -> LLMSingleRequest:
        prompt = SEGMENT_CAPTION_PROMPT.format(
            asr=artifact.related_asr
        )
        local_video_path = await fetch_object_from_s3(
            artifact.related_video_minio_url,
            self.visitor.minio_client,
            suffix=artifact.related_video_extension,
        )

        try:
            image_per_segments = cast(int, self.kwargs.get('image_per_segments'))
            image_encode = extract_images(local_video_path, artifact.start_frame, artifact.end_frame, image_per_segments)
            request = LLMSingleRequest(
                prompt=prompt,
                image_base64=image_encode
            )
            return request
        finally:
            cleanup_temp_file(local_video_path)

    async def execute(self, input_data: list[SegmentCaptionArtifact], client: BaseServiceClient | None| BaseMilvusClient) -> AsyncIterator[tuple[SegmentCaptionArtifact, str | None]]:

        logger = get_run_logger()

        assert client is not None, "The execution required client service"
        assert isinstance(client, BaseServiceClient)

        batch: list[SegmentCaptionArtifact] = []
        batches: list[list[SegmentCaptionArtifact]] = []

        bs = cast(int, self.kwargs.get('batch_size'))
        logger.info("Starting segment caption execution for %d artifacts", len(input_data))

        while input_data:
            artifact = input_data.pop(0)

            exists = await artifact.accept_check_exist(self.visitor)
            if exists:
                yield artifact, None
                continue
            batch.append(artifact)
            
            if len(batch) == bs:
                batches.append(batch[:])
                batch.clear()
        if batch:
            batches.append(batch[:])

        for batch in tqdm(batches, desc='Calling LLM...'):
            single_request = [
                await self._process_single_artifact(artifact=seg_artifact) for seg_artifact in batch
            ]
            request = LLMRequest(
                llm_requests=single_request,
                metadata={}
            )
            response = await client.make_request(
                method='POST',
                endpoint=client.inference_endpoint,
                request_data=request
            )
            parsed = LLMResponse.model_validate(response)

            for artifact, response in zip(batch,parsed.responses):
                caption = response.answer
                yield artifact, caption
    
            
    async def postprocess(self, output_data: tuple[SegmentCaptionArtifact, str | None]) -> SegmentCaptionArtifact:

        artifact, caption = output_data
        if caption is None:
            return artifact

        await artifact.accept_upload(self.visitor, caption)
        return artifact
    

    
