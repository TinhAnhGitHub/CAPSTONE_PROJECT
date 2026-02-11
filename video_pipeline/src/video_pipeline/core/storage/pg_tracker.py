from typing import TYPE_CHECKING, BinaryIO
from datetime import datetime
from video_pipeline.core.client.storage.pg import PostgresClient, ArtifactMetadata
from video_pipeline.core.client.storage.minio import MinioStorageClient


if TYPE_CHECKING:
    from video_pipeline.core.artifact import BaseArtifact


class ArtifactPersistentVisitor:
    def __init__(
        self, minio_client: MinioStorageClient, postgres_client: PostgresClient
    ):
        self.minio_client = minio_client
        self.postgres_client = postgres_client

    async def _check_exist(self, artifact: "BaseArtifact") -> bool:
        try:
            artifact_id = artifact.artifact_id  # type:ignore
            metadata = await self.postgres_client.get_artifact(artifact_id)
            if not metadata:
                return False

            return True
        except Exception as e:
            raise e

    async def visit_artifact(
        self, artifact: "BaseArtifact", upload_to_minio: BinaryIO | None = None
    ):
        metadata = artifact.metadata or {}
        metadata.update(**artifact.model_dump(mode="json"))
        if upload_to_minio:
            assert artifact.object_name, "If uploaded binary file to minio, please overide the method construct_object_name()"

            self.minio_client.upload_fileobj(
                bucket=artifact.user_id,
                object_name=artifact.object_name,
                file_obj=upload_to_minio,
            )

        artifact_metadata = ArtifactMetadata(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            minio_url=artifact.minio_url_path,
            user_id=artifact.user_id,
            lineage_parents=artifact.lineage_parents,
            created_at=datetime.now(),
            artifact_metadata=metadata,
        )
        await self.postgres_client.save_artifact(artifact_metadata)
