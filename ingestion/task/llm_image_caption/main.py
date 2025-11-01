from __future__  import annotations
from typing import  AsyncIterator, cast, Literal
from prefect import task
from core.pipeline.base_task import BaseTask
from core.clients.base import BaseServiceClient, BaseMilvusClient
from tqdm.asyncio import tqdm
from pydantic import BaseModel
from core.artifact.persist import  ArtifactPersistentVisitor 
from core.artifact.schema import ImageArtifact, ImageCaptionArtifact
from prefect_agent.service_llm.schema import LLMRequest, LLMResponse

from task.common.util import fetch_object_from_s3

from .util import  encode_image_base64
from .prompt import IMAGE_CAPTION
from core.config.logging import run_logger


class ImageCaptionSettings(BaseModel):
    model_name: str
    device: Literal['cpu', 'cuda']


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
            kwargs=kwargs
        )

    async def preprocess(self, input_data: list[ImageArtifact]) -> list[ImageCaptionArtifact]:
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
        
        run_logger.debug(f"Input data in image captioning: {result=}")
        return result

    async def execute(self, input_data: list[ImageCaptionArtifact], client: BaseServiceClient | None | BaseMilvusClient ) -> AsyncIterator[tuple[ImageCaptionArtifact, str | None]]:
    
        assert client is not None, "The execution required client service"
        assert isinstance(client, BaseServiceClient)



        for artifact in tqdm(input_data, desc="Processing data"):
            exists = await artifact.accept_check_exist(self.visitor)
            if exists:
                yield artifact , None
                continue
        
            prompt = IMAGE_CAPTION
            local_video_path = await fetch_object_from_s3(artifact.image_minio_url, self.visitor.minio_client, suffix=artifact.extension)
            image_encode = encode_image_base64(local_video_path)
            request = LLMRequest(
                prompt=prompt,
                image_base64=[image_encode],
                metadata={}
            )
            response = await client.make_request(
                method='POST',
                endpoint=client.inference_endpoint,
                request_data=request
            )

            parse = LLMResponse.model_validate(response)
            caption = parse.answer

            yield artifact, caption
    
    async def postprocess(self, output_data: tuple[ImageCaptionArtifact, str | None]) -> ImageCaptionArtifact:

        artifact, caption = output_data
        if caption is None:
            return artifact

        await artifact.accept_upload(self.visitor, caption)
        return artifact