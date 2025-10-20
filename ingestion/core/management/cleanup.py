from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from sqlalchemy import select, delete
from loguru import logger
from core.clients.base import BaseMilvusClient, MilvusCollectionConfig
from core.clients import milvus_client as _milvus_clients  
from core.pipeline.tracker import ArtifactTracker, ArtifactSchema, ArtifactLineageSchema
from core.storage import StorageClient, StorageError
from task.common.util import parse_s3_url
from core.app_state import AppState
from core.clients.milvus_client import ImageEmbeddingMilvusClient, TextCaptionEmbeddingMilvusClient, SegmentCaptionEmbeddingMilvusClient
class DeletionResult(BaseModel):
    """Result of a deletion operation."""
    success: bool
    video_id: str
    metadata: dict[str, Any]

class ArtifactDeleter:
    def __init__(
            self, 
            tracker: ArtifactTracker, 
            storage: StorageClient, 
            image_client: ImageEmbeddingMilvusClient,
            text_cap_client: TextCaptionEmbeddingMilvusClient,
            text_seg_client: SegmentCaptionEmbeddingMilvusClient,):
        
        self.tracker = tracker
        self.storage = storage
        self.image_client = image_client
        self.text_cap_client = text_cap_client
        self.text_seg_client =  text_seg_client
        self._state = AppState()

    async def get_all_descendants(
        self,
        session: Any,
        parent_id: str,
        visited: set[str] | None = None
    ) -> set[str]:
        if visited is None:
            visited = set()
        
        if parent_id in visited:
            return visited
        
        visited.add(parent_id)

        query = select(ArtifactSchema.artifact_id).where(
            ArtifactSchema.parent_artifact_id == parent_id
        )
        result = await session.execute(query)
        child_ids = {row[0] for row in result.all()}

        for child_id in child_ids:
            descendants = await self.get_all_descendants(
                session, child_id, visited
            )
            visited.update(descendants)
        
        return visited

    async def delete_by_related_video_id(self, related_video_id: str):
        clients: list[tuple[str, BaseMilvusClient ]] = [
              ("image_embedding", self.image_client),
              ("text_caption_embedding", self.text_cap_client),
              ("segment_caption_embedding", self.text_seg_client),
        ]
        errors: list[str] = []
        per_collection: dict[str, int] = {}
        total = 0

        for collection_name, client in clients:
            try:
                if not await client.has_user_collection():
                    per_collection[collection_name] = 0
                    continue

                filter_expr = (
                    f'related_video_id == "{related_video_id}"'
                )
                deleted = await client.delete_by_filter(filter_expr)
                per_collection[collection_name] = deleted
                total += deleted

            except Exception as e:
                err = f"{client.__name__}: {e}"
                logger.exception("milvus_dynamic_delete_error", error=str(e))
                errors.append(err)
        
        status_dict = {
            'success': len(errors) == 0,
            'related_video_id': related_video_id,
            'per_collection_deleted': per_collection,
            'total_deleted': total,
            'errors': errors
        }
        return status_dict

    
    async def delete_video_cascade(self, video_id: str) -> DeletionResult:
        errors: list[str] = []
        deleted_artifacts = 0
        deleted_lineage = 0
        deleted_minio = 0

        try:
            async with self.tracker.get_session() as session:
                # Verify video exists
                video_query = select(ArtifactSchema).where(
                    ArtifactSchema.artifact_id == video_id
                )
                video_result = await session.execute(video_query)
                video_artifact = video_result.scalar_one_or_none()
                
                if not video_artifact:
                    raise RuntimeError(
                        f"Video not found: {video_id}"
                    )

                all_artifact_ids = await self.get_all_descendants(
                    session, video_id
                )

                all_artifact_ids.add(video_id)
                logger.info(
                    f"Found {len(all_artifact_ids)} artifacts to delete for video {video_id}"
                )

                artifacts_query = select(ArtifactSchema).where(
                    ArtifactSchema.artifact_id.in_(all_artifact_ids)
                )
                artifacts_result = await session.execute(artifacts_query)
                artifacts = artifacts_result.scalars().all()

                for artifact in artifacts:
                    try:
                        bucket, object_key = parse_s3_url(artifact.minio_url)
                        
                        if self.storage.object_exists(bucket, object_key):
                            self.storage.client.remove_object(bucket, object_key)
                            deleted_minio += 1
                            logger.debug(f"Deleted MinIO object: {artifact.minio_url}")
                    except StorageError as e:
                        error_msg = f"Failed to delete MinIO object {artifact.minio_url}: {e}"
                        logger.warning(error_msg)
                        errors.append(error_msg)
                    except Exception as e:
                        error_msg = f"Unexpected error deleting {artifact.minio_url}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                
                lineage_delete = delete(ArtifactLineageSchema).where(
                    (ArtifactLineageSchema.parent_artifact_id.in_(all_artifact_ids)) |
                    (ArtifactLineageSchema.child_artifact_id.in_(all_artifact_ids))
                )
                lineage_result = await session.execute(lineage_delete)
                deleted_lineage = lineage_result.rowcount or 0

                artifacts_delete = delete(ArtifactSchema).where(
                    ArtifactSchema.artifact_id.in_(all_artifact_ids)
                )
                artifacts_result = await session.execute(artifacts_delete)
                deleted_artifacts = artifacts_result.rowcount or 0

                await session.commit()


                meta_status = await self.delete_by_related_video_id(related_video_id=video_id)
                logger.info(
                    f"Successfully deleted video {video_id}: "
                    f"{deleted_artifacts} artifacts, "
                    f"{deleted_lineage} lineage records, "
                    f"{deleted_minio} MinIO objects"
                    f"and milvus info: {meta_status}"
                )

                return_metadata = {
                    'deleted_artifacts': deleted_artifacts,
                    'deleted_lineage': deleted_lineage,
                    'deleted_minio_objects': deleted_minio,
                    'milvus_delete': meta_status,
                    'errors': errors
                }

                return DeletionResult(
                    success=len(errors) == 0,
                    video_id=video_id,
                    metadata=return_metadata
                )
            

        except Exception as e:
            logger.exception(f"Failed to delete video {video_id}: {e}")
            raise RuntimeError(
                f"Deletion failed: {str(e)}"
            )
    

    async def delete_stage_artifacts(
        self,
        video_id: str, 
        artifact_type: str
    )->DeletionResult:
        errors: list[str] = []
        deleted_artifacts = 0
        deleted_lineage = 0
        deleted_minio = 0

        try:
            async with self.tracker.get_session() as session:
                video_query = select(ArtifactSchema).where(
                ArtifactSchema.artifact_id == video_id
            )
            video_result = await session.execute(video_query)
            video_artifact = video_result.scalar_one_or_none()
            
            if not video_artifact:
                raise RuntimeError(f"Video not found: {video_id}")

            all_descendants = await self.get_all_descendants(session, video_id)
            all_descendants.add(video_id)

            stage_query = select(ArtifactSchema).where(
                ArtifactSchema.artifact_id.in_(all_descendants),
                ArtifactSchema.artifact_type == artifact_type
            )
            stage_result = await session.execute(stage_query)
            stage_artifacts = stage_result.scalars().all()

            if not stage_artifacts:
                logger.info(
                    f"No artifacts of type '{artifact_type}' found for video {video_id}"
                )
                return_metadata = {
                    'deleted_artifacts': 0,
                    'deleted_lineage': 0,
                    'deleted_minio_objects': 0,
                    'errors':[]
                }
                return DeletionResult(
                    success=True,
                    video_id=video_id,
                    metadata=return_metadata
                )
            

            artifacts_to_delete = set()
            for artifact in stage_artifacts:
                descendants = await self.get_all_descendants(
                    session, artifact.artifact_id
                )
                artifacts_to_delete.update(descendants)
                artifacts_to_delete.add(artifact.artifact_id)
            
            artifacts_query = select(ArtifactSchema).where(
                ArtifactSchema.artifact_id.in_(artifacts_to_delete)
            )
            artifacts_result = await session.execute(artifacts_query)
            artifacts = artifacts_result.scalars().all()

            for artifact in artifacts:
                try:
                    bucket, object_key = parse_s3_url(artifact.minio_url)
                    
                    if self.storage.object_exists(bucket, object_key):
                        self.storage.client.remove_object(bucket, object_key)
                        deleted_minio += 1
                        logger.debug(f"Deleted MinIO object: {artifact.minio_url}")
                except StorageError as e:
                    error_msg = f"Failed to delete MinIO object {artifact.minio_url}: {e}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                except Exception as e:
                    error_msg = f"Unexpected error deleting {artifact.minio_url}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            lineage_delete = delete(ArtifactLineageSchema).where(
                (ArtifactLineageSchema.parent_artifact_id.in_(artifacts_to_delete)) |
                (ArtifactLineageSchema.child_artifact_id.in_(artifacts_to_delete))
            )
            lineage_result = await session.execute(lineage_delete)
            deleted_lineage = lineage_result.rowcount or 0

            artifacts_delete = delete(ArtifactSchema).where(
                ArtifactSchema.artifact_id.in_(artifacts_to_delete)
            )
            artifacts_result = await session.execute(artifacts_delete)
            deleted_artifacts = artifacts_result.rowcount or 0

            await session.commit()
            meta_status = await self.delete_by_related_video_id(related_video_id=video_id)
            return_metadata = {
                    'deleted_artifacts': deleted_artifacts,
                    'deleted_lineage': deleted_lineage,
                    'deleted_minio_objects': deleted_minio,
                    'milvus_delete': meta_status,
                    'errors': errors
                }

            return DeletionResult(
                success=len(errors) == 0,
                video_id=video_id,
                metadata=return_metadata
            )
        except Exception as e:
            logger.exception(
                f"Failed to delete stage '{artifact_type}' for video {video_id}: {e}"
            )
            raise RuntimeError(f"Deletion failed: {str(e)}")




