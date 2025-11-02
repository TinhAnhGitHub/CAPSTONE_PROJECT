from __future__ import annotations
from typing import AsyncIterator, Literal
from core.pipeline.base_task import BaseTask
from core.artifact.persist import ArtifactPersistentVisitor  
from core.artifact.schema import AutoshotArtifact, VideoArtifact
from prefect_agent.service_autoshot.schema import AutoShotRequest, AutoShotResponse
from pydantic import BaseModel
from core.clients.base import BaseServiceClient, BaseMilvusClient
from prefect.logging import get_run_logger

class AutoshotSettings(BaseModel):
    model_name: str
    device: Literal['cuda', 'cpu']
    
class AutoshotProcessingTask(BaseTask[list[VideoArtifact], AutoshotArtifact]):
    
    def __init__(
        self, 
        artifact_visitor: ArtifactPersistentVisitor,
        **kwargs
    ):
        super().__init__(
            name=AutoshotProcessingTask.__name__,
            visitor=artifact_visitor,
             **kwargs
        )
        self.name = 'autoshot'
    
    async def preprocess(self, input_data: list[VideoArtifact]) -> list[AutoshotArtifact]:
        """
        Preprocess the video artifact
        1. Prepare the model
        """

        list_autoshot_artifact = []
        for video_art in input_data:
            autoshot_art = AutoshotArtifact(
                related_video_id=video_art.artifact_id,
                related_video_minio_url=video_art.minio_url_path,
                related_video_extension=video_art.video_extension,
                task_name=self.name,
                user_bucket=video_art.user_bucket,
                artifact_type=AutoshotArtifact.__name__,
                related_video_fps=video_art.fps
            )   
            list_autoshot_artifact.append(autoshot_art)
        
        return list_autoshot_artifact
    

    async def execute(self, input_data: list[AutoshotArtifact], client: BaseServiceClient | None | BaseMilvusClient) -> AsyncIterator[tuple[AutoshotArtifact ,list[tuple[int,int]] | None]]:

        assert client is not None, "The execution required client service"
        assert isinstance(client, BaseServiceClient)
        logger = get_run_logger()
        logger.info("Starting autoshot execution for %d artifacts", len(input_data))

        for artifact in input_data:
            exist = await artifact.accept_check_exist(self.visitor)
            if exist:
                logger.debug(
                    "Autoshot artifact already exists for video %s; skipping inference",
                    artifact.related_video_id,
                )
            
                yield artifact, None
                continue
            
            logger.info(
                "Submitting autoshot request for video %s at %s",
                artifact.related_video_id,
                artifact.related_video_minio_url,
            )
            request = AutoShotRequest(
                s3_minio_url=artifact.related_video_minio_url,
                metadata={}
            )
            response = await client.make_request(
                method='POST',
                endpoint=client.inference_endpoint,
                request_data=request,
            )
            parsed = AutoShotResponse.model_validate(response)
            logger.info(
                "Received autoshot response for video %s with %d scenes",
                artifact.related_video_id,
                len(parsed.scenes),
            )
            yield artifact, parsed.scenes
            
        
    
    async def postprocess(self, output_data: tuple[AutoshotArtifact ,list[tuple[int,int]] | None]) -> AutoshotArtifact:
        artifact, scenes = output_data
        
        if scenes is None:
            return artifact
        await artifact.accept_upload(self.visitor, scenes)
        return artifact











        

        







        
