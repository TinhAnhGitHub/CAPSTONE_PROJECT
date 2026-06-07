from typing import TYPE_CHECKING, Any, BinaryIO
from datetime import datetime
from video_pipeline.core.client.storage.pg import PostgresClient, ArtifactMetadata
from video_pipeline.core.client.storage.minio import MinioStorageClient

if TYPE_CHECKING:
    from video_pipeline.core.artifact import BaseArtifact


def sanitize_metadata_value(value: Any) -> Any:
    """Remove PostgreSQL-incompatible null characters from nested metadata values."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {
            sanitize_metadata_value(key): sanitize_metadata_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_metadata_value(item) for item in value)
    return value


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
        metadata = sanitize_metadata_value(metadata)
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
