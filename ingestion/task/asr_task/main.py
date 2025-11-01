from __future__ import annotations
from typing import AsyncIterator, Literal, cast  
from prefect import task
from pydantic import BaseModel
from core.pipeline.base_task import BaseTask
from core.artifact.persist import ArtifactPersistentVisitor
from core.artifact.schema import ASRArtifact, VideoArtifact
from prefect_agent.service_asr.core.schema import ASRInferenceRequest, ASRInferenceResponse
from core.clients.base import BaseMilvusClient, BaseServiceClient



class ASRProcessingTask(BaseTask[list[VideoArtifact], ASRArtifact]):
    def __init__(
        self, 
        artifact_visitor: ArtifactPersistentVisitor,
        **kwargs
    ):
        super().__init__(
            name=ASRProcessingTask.__name__,
            visitor=artifact_visitor,
            kwargs=kwargs
        )
        self.name = 'asr'

    async def preprocess(self, input_data: list[VideoArtifact]) -> list[ASRArtifact]:
        """
        Preprocess the video artifact
        1. Prepare the model
        """

        list_autoshot_artifact = []

        for video_art in input_data:
            autoshot_art = ASRArtifact(
                related_video_id=video_art.artifact_id,
                related_video_minio_url=video_art.minio_url_path,
                related_video_extension=video_art.video_extension,
                task_name=self.name,
                user_bucket=video_art.user_bucket,
                related_video_fps=video_art.fps,
                artifact_type=ASRArtifact.__name__
            )                
            list_autoshot_artifact.append(autoshot_art)
        return list_autoshot_artifact

    async def execute(self, input_data: list[ASRArtifact], client: BaseServiceClient| BaseMilvusClient | None) -> AsyncIterator[tuple[ASRArtifact, dict|None ]]:

        assert client is not None, "The execution required client service"
        assert isinstance(client, BaseServiceClient)

        
        for artifact in input_data:
            exist = await artifact.accept_check_exist(self.visitor)
            

            if exist:
                yield artifact, None
                continue
            
            request = ASRInferenceRequest(
                video_minio_url=artifact.related_video_minio_url,
                metadata={}
            )


            response = await client.make_request(
                method='POST',
                endpoint=client.inference_endpoint,
                request_data=request,
            )
            parsed = ASRInferenceResponse.model_validate(response)
            result_asr = parsed.result.model_dump(mode='json')
            yield artifact, result_asr
    
    async def postprocess(self, output_data: tuple[ASRArtifact, dict|None ]) -> ASRArtifact:        
        artifact, data = output_data
        if data is None:
            return artifact
        
        await artifact.accept_upload(self.visitor, data)
        return artifact

    
    

