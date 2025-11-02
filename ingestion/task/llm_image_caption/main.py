from __future__ import annotations

from typing import AsyncIterator, Literal, cast

from core.artifact.persist import ArtifactPersistentVisitor
from core.artifact.schema import ImageArtifact, ImageCaptionArtifact
from core.clients.base import BaseMilvusClient, BaseServiceClient
from core.pipeline.base_task import BaseTask
from prefect.logging import get_run_logger
from prefect_agent.service_llm.schema import LLMRequest, LLMResponse, LLMSingleRequest
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from task.common.util import cleanup_temp_file, fetch_object_from_s3

from .prompt import IMAGE_CAPTION
from .util import encode_image_base64


class ImageCaptionLLMTask(BaseTask[
    list[ImageArtifact], ImageCaptionArtifact,
]):
    def __init__(
        self,
        artifact_visitor: ArtifactPersistentVisitor,
        **kwargs,
    ):
        super().__init__(
            name=ImageCaptionLLMTask.__name__,
            visitor=artifact_visitor,
             **kwargs
        )

    async def preprocess(self, input_data: list[ImageArtifact]) -> list[ImageCaptionArtifact]:
        logger = get_run_logger()
        result = []
        for img_artifact in input_data:
            fps = img_artifact.related_video_fps
            timestamp =img_artifact.timestamp
            img_cap_artifact = ImageCaptionArtifact(
                time_stamp=timestamp,
                related_video_fps=fps,
                frame_index=img_artifact.frame_index,
                user_bucket=img_artifact.user_bucket,
                image_minio_url=img_artifact.minio_url_path,
                extension=img_artifact.extension,
                image_id=img_artifact.artifact_id,
                artifact_type=ImageArtifact.__name__,
                related_video_id=img_artifact.related_video_id
            )
            result.append(img_cap_artifact)
        
        logger.debug("Prepared %d image caption artifacts", len(result))
        return result

    async def _process_single_artifact(self, artifact: ImageCaptionArtifact) -> LLMSingleRequest:
        prompt = IMAGE_CAPTION
        local_image_path = await fetch_object_from_s3(
            artifact.image_minio_url,
            self.visitor.minio_client,
            suffix=artifact.extension,
        )
        try:
            image_encode = encode_image_base64(local_image_path)

            request = LLMSingleRequest(
                prompt=prompt,
                image_base64=[image_encode]
            )
            return request
        finally:
            cleanup_temp_file(local_image_path)
    async def execute(self, input_data: list[ImageCaptionArtifact], client: BaseServiceClient | None | BaseMilvusClient ) -> AsyncIterator[tuple[ImageCaptionArtifact, str | None]]:
        logger = get_run_logger()
        logger.info("Starting image caption execution for %d artifacts", len(input_data))
        assert client is not None, "The execution required client service"
        assert isinstance(client, BaseServiceClient)

        batch: list[ImageCaptionArtifact] = []
        batches: list[list[ImageCaptionArtifact]] = []

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

    
    async def postprocess(self, output_data: tuple[ImageCaptionArtifact, str | None]) -> ImageCaptionArtifact:

        artifact, caption = output_data
        if caption is None:
            return artifact

        await artifact.accept_upload(self.visitor, caption)
        return artifact
