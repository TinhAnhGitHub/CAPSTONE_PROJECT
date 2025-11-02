from __future__ import annotations

from core.storage import StorageClient
from core.pipeline.tracker import ArtifactTracker, ArtifactMetadata
from typing import BinaryIO, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .schema import (
        BaseArtifact,
        VideoArtifact,
        ImageArtifact,
        AutoshotArtifact,
        ASRArtifact,
        SegmentCaptionArtifact,
        ImageCaptionArtifact,
        ImageEmbeddingArtifact,
        TextCaptionEmbeddingArtifact,
        TextCapSegmentEmbedArtifact,
    )
class ArtifactPersistentVisitor:
    def __init__(
        self,
        minio_client: StorageClient,
        tracker: ArtifactTracker
    ):
        self.minio_client = minio_client
        self.tracker = tracker
    

    async def _check_exist(self, artifact: "BaseArtifact", bucket_name: str, check_minio:bool=True) -> bool:
        try:
            artifact_id = artifact.artifact_id
            metadata = await self.tracker.get_artifact(artifact_id)
            if not metadata:
                return False

            object_key = artifact.object_key
            if check_minio:
                object_returned = self.minio_client.get_object(
                    bucket=bucket_name,
                    object_name=object_key,
                )
                return object_returned is not None
            return True


        except Exception as e:
            raise e


    async def visit_video(self, artifact: "VideoArtifact", upload_file:dict):

        artifact_metadata = ArtifactMetadata(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            minio_url=artifact.video_minio_url,
            user_id=artifact.user_bucket,
            parent_artifact_id=None,
            task_name=artifact.task_name,
            created_at=datetime.now(),
            artifact_metadata=upload_file
        )
        print(f"{artifact_metadata=}")

        
        await self.tracker.save_artifact(artifact_metadata)

    async def visit_segments(self, artifact: "AutoshotArtifact", upload_file: list):
        
    
        object_key = artifact.object_key

        minio_url= self.minio_client.put_json(
            bucket=artifact.user_bucket,
            object_name=object_key,
            payload={
                "segments": upload_file,
                **artifact.model_dump(mode='json')
            }
        )
        artifact_metadata = ArtifactMetadata(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            minio_url=minio_url,
            user_id=artifact.user_bucket,
            parent_artifact_id=artifact.related_video_id,
            task_name=artifact.task_name,
            created_at=datetime.now(),
            artifact_metadata={}
        )
        await self.tracker.save_artifact(artifact_metadata)

    


    async def visit_asr(self, artifact: "ASRArtifact", upload_file: dict):
        object_key = artifact.object_key
        minio_url = self.minio_client.put_json(
            bucket=artifact.user_bucket,
            object_name=object_key,
            payload=upload_file
        )

        artifact_metadata = ArtifactMetadata(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            minio_url=minio_url,
            user_id=artifact.user_bucket,
            parent_artifact_id=artifact.related_video_id,
            task_name=artifact.task_name,
            created_at=datetime.now(),
            artifact_metadata={}
        )
        await self.tracker.save_artifact(artifact_metadata)


    async def visit_image(self, artifact: "ImageArtifact", upload_file: BinaryIO):

        object_key = artifact.object_key
        self.minio_client.upload_fileobj(
            bucket=artifact.user_bucket,
            object_name=object_key,
            file_obj=upload_file,
            content_type=artifact.content_type,
            metadata=artifact.metadata
        )

        artifact_metadata = ArtifactMetadata(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            minio_url=artifact.minio_url_path,
            parent_artifact_id=artifact.autoshot_artifact_id,
            task_name='image processing',
            user_id=artifact.user_bucket,
            artifact_metadata={}
        )
        await self.tracker.save_artifact(artifact_metadata)

    
    async def visit_segment_caption(self, artifact: "SegmentCaptionArtifact", upload_caption: str):
        
        object_key = artifact.object_key

        payload = artifact.model_dump(mode='json')
        payload.update(
            {
                'caption': upload_caption
            }
        )
        artifact_metadata = ArtifactMetadata(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            minio_url=artifact.minio_url_path,
            parent_artifact_id=artifact.autoshot_artifact_id,
            task_name='Segment caption',
            user_id=artifact.user_bucket,
            artifact_metadata={}
            
        )
        await self.tracker.save_artifact(artifact_metadata)

        self.minio_client.put_json(
            bucket=artifact.user_bucket,
            object_name=object_key,
            payload=payload
        )

    async def visit_image_caption(self, artifact: "ImageCaptionArtifact", upload_caption: str ):
        object_key = artifact.object_key
        payload = artifact.model_dump(mode='json')
        payload.update(
            {
                'caption': upload_caption
            }
        )
        artifact_metadata = ArtifactMetadata(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            minio_url=artifact.minio_url_path,
            parent_artifact_id=artifact.image_id,
            task_name='image caption',
            user_id=artifact.user_bucket,
            artifact_metadata={}
        )
        await self.tracker.save_artifact(artifact_metadata)

        self.minio_client.put_json(
            bucket=artifact.user_bucket,
            object_name=object_key,
            payload=payload
        )

    async def visit_image_embedding(self, artifact: "ImageEmbeddingArtifact", upload_file: BinaryIO):
        object_key = artifact.object_key

        minio_url = self.minio_client.upload_fileobj(
            bucket=artifact.user_bucket,
            object_name=object_key,
            file_obj=upload_file,
            content_type="application/json",
            metadata={}
        )

        artifact_metadata = ArtifactMetadata(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            minio_url=artifact.minio_url_path,
            parent_artifact_id=artifact.image_id,
            task_name='image embedding',
            user_id=artifact.user_bucket,
            artifact_metadata={}
        )
        await self.tracker.save_artifact(artifact_metadata)


    async def visit_image_caption_embedding(self, artifact: "TextCaptionEmbeddingArtifact", upload_file: BinaryIO):
        object_key = artifact.object_key
        minio_url = self.minio_client.upload_fileobj(
            bucket=artifact.user_bucket,
            object_name=object_key,
            file_obj=upload_file,
            content_type="application/json",
            metadata={}
        )

        artifact_metadata = ArtifactMetadata(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            minio_url=artifact.minio_url_path,
            parent_artifact_id=artifact.caption_id,
            task_name='image embedding',
            user_id=artifact.user_bucket,
            artifact_metadata={}
        )
        await self.tracker.save_artifact(artifact_metadata)
        return minio_url
    
    async def visit_segment_caption_embedding(self, artifact: "TextCapSegmentEmbedArtifact", upload_file: BinaryIO):
        object_key = artifact.object_key
        minio_url = self.minio_client.upload_fileobj(
            bucket=artifact.user_bucket,
            object_name=object_key,
            file_obj=upload_file,
            content_type="application/json",
            metadata={}
        )

        artifact_metadata = ArtifactMetadata(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            minio_url=artifact.minio_url_path,
            parent_artifact_id=artifact.segment_cap_id,
            task_name='image embedding',
            user_id=artifact.user_bucket,
            artifact_metadata={}            
        )
        await self.tracker.save_artifact(artifact_metadata)
        return minio_url
    
