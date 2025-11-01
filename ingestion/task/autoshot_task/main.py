from __future__ import annotations
from typing import AsyncIterator, Literal
from core.pipeline.base_task import BaseTask
from core.artifact.persist import ArtifactPersistentVisitor  
from core.artifact.schema import AutoshotArtifact, VideoArtifact
from prefect_agent.service_autoshot.schema import AutoShotRequest, AutoShotResponse
from pydantic import BaseModel
from core.clients.base import BaseServiceClient, BaseMilvusClient
from core.config.logging import run_logger





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
            kwargs=kwargs
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
        
        for artifact in input_data:
            exist = await artifact.accept_check_exist(self.visitor)
            run_logger.debug(f"Artifact: {artifact.model_dump(mode='json')} exists: {exist}")
            
            if exist:
                yield artifact, None
                continue
            
            run_logger.debug(f"Calling request to autoshot with s3 minio: {artifact.related_video_minio_url}")
            request = AutoShotRequest(
                s3_minio_url=artifact.related_video_minio_url,
                metadata={}
            )
            run_logger.debug(f"Request: {request.model_dump(mode='json')}")
            response = await client.make_request(
                method='POST',
                endpoint=client.inference_endpoint,
                request_data=request,
            )
            run_logger.debug(f"Response: {response}")
            parsed = AutoShotResponse.model_validate(response)
            yield artifact, parsed.scenes
            
        
    
    async def postprocess(self, output_data: tuple[AutoshotArtifact ,list[tuple[int,int]] | None]) -> AutoshotArtifact:
        artifact, scenes = output_data
        
        if scenes is None:
            return artifact
        await artifact.accept_upload(self.visitor, scenes)
        return artifact











        

        







        
