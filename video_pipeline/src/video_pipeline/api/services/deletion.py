from __future__ import annotations

from typing import Any

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from video_pipeline.config import get_settings
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.schema import ArtifactSchema, ArtifactLineageSchema


class VideoDeletionService:
    """Service to delete all artifacts associated with a video_id."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def _get_postgres_session(self) -> AsyncSession:
        engine = create_async_engine(
            url=self.settings.postgres.connection_string,
            echo=False,
            poolclass=NullPool,
        )
        sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        return sessionmaker()

    def _get_minio_client(self) -> MinioStorageClient:
        """Get a MinIO client."""
        return MinioStorageClient(
            endpoint=self.settings.minio.endpoint,
            access_key=self.settings.minio.access_key,
            secret_key=self.settings.minio.secret_key,
            secure=self.settings.minio.secure,
        )

    def _get_qdrant_client(self) -> AsyncQdrantClient:
        """Get a Qdrant client."""
        return AsyncQdrantClient(
            host=self.settings.qdrant.host,
            port=self.settings.qdrant.port,
        )

    async def _get_artifacts_for_video(self, video_id: str, session: AsyncSession) -> list[dict[str, Any]]:
        """Retrieve all artifacts related to a video_id from PostgreSQL.

        Returns:
            List of artifact metadata dicts containing object_name, user_id, etc.
        """
        stmt = select(ArtifactSchema).where(
            ArtifactSchema.artifact_metadata["related_video_id"].as_string() == video_id
        )
        result = await session.execute(stmt)
        artifacts = result.scalars().all()

        stmt2 = select(ArtifactSchema).where(ArtifactSchema.artifact_id == video_id)
        result2 = await session.execute(stmt2)
        video_artifact = result2.scalars().first()

        if video_artifact and video_artifact not in artifacts:
            artifacts = list(artifacts) + [video_artifact]

        return [a.artifact_metadata for a in artifacts if a.artifact_metadata]

    async def delete_video(self, video_id: str) -> dict[str, Any]:
        """Delete all artifacts associated with a video_id.

        Process:
        1. Query PostgreSQL for all artifacts with related_video_id or artifact_id == video_id
        2. Extract object_name from artifact metadata and delete from MinIO
        3. Delete lineage records from PostgreSQL
        4. Delete artifacts from PostgreSQL
        5. Delete vectors from Qdrant

        Args:
            video_id: The video ID to delete all artifacts for

        Returns:
            Dict with deletion results from each storage backend
        """
        logger.info(f"[VideoDeletion] Starting deletion for video_id={video_id}")

        results = {
            "video_id": video_id,
            "postgres": {"artifacts": 0, "lineage": 0},
            "minio": {"objects_deleted": 0, "errors": []},
            "qdrant": {},
            "summary": {"success": False},
        }

        session = await self._get_postgres_session()

        try:
            artifact_metadatas = await self._get_artifacts_for_video(video_id, session)

            if not artifact_metadatas:
                logger.info(f"[VideoDeletion] No artifacts found for video_id={video_id}")
                results["summary"]["success"] = True
                return results

            artifact_ids = [m.get("artifact_id") for m in artifact_metadatas if m.get("artifact_id")]

            # Step 2: Delete from MinIO using object_name from metadata
            minio_client = self._get_minio_client()
            objects_deleted = 0
            minio_errors = []

            for metadata in artifact_metadatas:
                object_name = metadata.get("object_name")
                user_id = metadata.get("user_id")

                if object_name and user_id:
                    try:
                        minio_client.delete_object(bucket=user_id, object_name=object_name)
                        objects_deleted += 1
                        logger.debug(f"[MinIO] Deleted {user_id}/{object_name}")
                    except Exception as e:
                        err_msg = f"{user_id}/{object_name}: {str(e)}"
                        minio_errors.append(err_msg)
                        logger.warning(f"[MinIO] Failed to delete {err_msg}")

            results["minio"]["objects_deleted"] = objects_deleted
            results["minio"]["errors"] = minio_errors

            lineage_stmt = delete(ArtifactLineageSchema).where(
                ArtifactLineageSchema.child_artifact_id.in_(artifact_ids)
            )
            lineage_result = await session.execute(lineage_stmt)
            lineage_count = lineage_result.rowcount

            parent_lineage_stmt = delete(ArtifactLineageSchema).where(
                ArtifactLineageSchema.parent_artifact_id.in_(artifact_ids)
            )
            parent_lineage_result = await session.execute(parent_lineage_stmt)
            lineage_count += parent_lineage_result.rowcount

            results["postgres"]["lineage"] = lineage_count

            artifact_stmt = delete(ArtifactSchema).where(
                ArtifactSchema.artifact_id.in_(artifact_ids)
            )
            artifact_result = await session.execute(artifact_stmt)
            artifact_count = artifact_result.rowcount

            results["postgres"]["artifacts"] = artifact_count

            await session.commit()

            logger.info(
                f"[Postgres] Deleted {artifact_count} artifacts and {lineage_count} lineage records"
            )

            qdrant_client = self._get_qdrant_client()
            collection_base = self.settings.qdrant.collection_base
            collections = [
                f"{collection_base}_image",
                f"{collection_base}_caption",
                f"{collection_base}_segment",
            ]

            for collection_name in collections:
                try:
                    exists = await qdrant_client.collection_exists(collection_name)
                    if not exists:
                        continue

                    filter_condition = Filter(
                        must=[
                            FieldCondition(
                                key="related_video_id",
                                match=MatchValue(value=video_id)
                            )
                        ]
                    )

                    await qdrant_client.delete(
                        collection_name=collection_name,
                        points_selector=filter_condition,
                        wait=True,
                    )
                    results["qdrant"][collection_name] = "deleted"
                    logger.info(f"[Qdrant] Deleted vectors from {collection_name}")

                except Exception as e:
                    results["qdrant"][collection_name] = f"error: {str(e)}"
                    logger.warning(f"[Qdrant] Error deleting from {collection_name}: {e}")

            await qdrant_client.close()

            results["summary"]["success"] = True
            results["summary"]["artifacts_deleted"] = artifact_count
            results["summary"]["objects_deleted"] = objects_deleted

        except Exception as e:
            logger.exception(f"[VideoDeletion] Error during deletion: {e}")
            results["error"] = str(e)
            results["summary"]["success"] = False

        finally:
            await session.close()

        logger.info(f"[VideoDeletion] Completed for video_id={video_id}: {results['summary']}")
        return results