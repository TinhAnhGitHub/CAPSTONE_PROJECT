from __future__ import annotations

from typing import Any

from arango.client import ArangoClient
from elasticsearch import AsyncElasticsearch
from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from video_pipeline.config import get_settings
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.schema import ArtifactSchema, ArtifactLineageSchema


ARANGO_VERTEX_COLLECTIONS = ["entities", "events", "micro_events", "communities"]
ARANGO_EDGE_COLLECTIONS = [
    "entity_relations",
    "event_sequences",
    "event_entities",
    "micro_event_sequences",
    "micro_event_parents",
    "micro_event_entities",
    "community_members",
    "event_communities",
]


class VideoDeletionService:

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
        return MinioStorageClient(
            endpoint=self.settings.minio.endpoint,
            access_key=self.settings.minio.access_key,
            secret_key=self.settings.minio.secret_key,
            secure=self.settings.minio.secure,
        )

    def _get_qdrant_client(self) -> AsyncQdrantClient:
        return AsyncQdrantClient(
            host=self.settings.qdrant.host,
            port=self.settings.qdrant.port,
        )

    def _get_arango_db(self):
        client = ArangoClient(hosts=self.settings.arango.host)
        return client.db(self.settings.arango.database)

    def _get_elasticsearch_client(self) -> AsyncElasticsearch:
        es = self.settings.elasticsearch
        url = f"http://{es.host}:{es.port}"
        return AsyncElasticsearch(url)

    async def _get_artifacts_for_video(self, video_id: str, session: AsyncSession) -> list[dict[str, Any]]:
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

    def _delete_from_arango(self, video_id: str) -> dict[str, Any]:
        result: dict = {}
        try:
            db = self._get_arango_db()

            for collection in ARANGO_VERTEX_COLLECTIONS:
                try:
                    query = f"FOR doc IN {collection} FILTER doc.video_id == @video_id REMOVE doc IN {collection}"
                    cursor = db.aql.execute(query, bind_vars={"video_id": video_id})
                    stats = cursor.statistics()  # type: ignore
                    count = stats.get("writes_executed", 0) if stats else 0
                    result[collection] = {"deleted": count}
                    logger.info(f"[ArangoDB] Deleted {count} documents from {collection}")
                except Exception as e:
                    result[collection] = {"error": str(e)}
                    logger.warning(f"[ArangoDB] Error deleting from {collection}: {e}")

            for collection in ARANGO_EDGE_COLLECTIONS:
                try:
                    query = f"FOR doc IN {collection} FILTER doc.video_id == @video_id REMOVE doc IN {collection}"
                    cursor = db.aql.execute(query, bind_vars={"video_id": video_id})
                    stats = cursor.statistics()  # type: ignore
                    count = stats.get("writes_executed", 0) if stats else 0
                    result[collection] = {"deleted": count}
                    logger.info(f"[ArangoDB] Deleted {count} edges from {collection}")
                except Exception as e:
                    result[collection] = {"error": str(e)}
                    logger.warning(f"[ArangoDB] Error deleting from {collection}: {e}")

        except Exception as e:
            logger.warning(f"[ArangoDB] Connection error: {e}")
            result["error"] = str(e)

        return result

    async def _delete_from_elasticsearch(self, video_id: str) -> dict[str, Any]:
        result: dict = {"deleted": 0}
        client = self._get_elasticsearch_client()

        try:
            resp = await client.delete_by_query(
                index=self.settings.elasticsearch.index_name,
                body={"query": {"term": {"video_id": video_id}}}
            )
            deleted = resp.get("deleted", 0)
            result["deleted"] = deleted
            logger.info(f"[Elasticsearch] Deleted {deleted} OCR documents")

        except Exception as e:
            logger.warning(f"[Elasticsearch] Error deleting documents: {e}")
            result["error"] = str(e)

        finally:
            await client.close()

        return result

    async def delete_video(self, video_id: str) -> dict[str, Any]:
        """Delete all artifacts associated with a video_id.

        Process:
        1. Query PostgreSQL for all artifacts with related_video_id or artifact_id == video_id
        2. Extract object_name from artifact metadata and delete from MinIO
        3. Delete lineage records from PostgreSQL
        4. Delete artifacts from PostgreSQL
        5. Delete vectors from Qdrant
        6. Delete KG data from ArangoDB
        7. Delete OCR documents from Elasticsearch

        Args:
            video_id: The video ID to delete all artifacts for

        Returns:
            Dict with deletion results from each storage backend
        """
        logger.info(f"[VideoDeletion] Starting deletion for video_id={video_id}")

        results : dict = {
            "video_id": video_id,
            "postgres": {"artifacts": 0, "lineage": 0},
            "minio": {"objects_deleted": 0, "errors": []},
            "qdrant": {},
            "arango": {},
            "elasticsearch": {"deleted": 0},
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

            # Delete lineage records
            lineage_stmt = delete(ArtifactLineageSchema).where(
                ArtifactLineageSchema.child_artifact_id.in_(artifact_ids)
            )
            lineage_result = await session.execute(lineage_stmt)
            lineage_count = lineage_result.rowcount #type:ignore

            parent_lineage_stmt = delete(ArtifactLineageSchema).where(
                ArtifactLineageSchema.parent_artifact_id.in_(artifact_ids)
            )
            parent_lineage_result = await session.execute(parent_lineage_stmt)
            lineage_count += parent_lineage_result.rowcount #type:ignore

            results["postgres"]["lineage"] = lineage_count

            artifact_stmt = delete(ArtifactSchema).where(
                ArtifactSchema.artifact_id.in_(artifact_ids)
            )
            artifact_result = await session.execute(artifact_stmt)
            artifact_count = artifact_result.rowcount #type:ignore

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
                f"{collection_base}_segment_caption",
                f"{collection_base}_audio_transcript",
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

            import asyncio
            results["arango"] = await asyncio.to_thread(self._delete_from_arango, video_id)

            results["elasticsearch"] = await self._delete_from_elasticsearch(video_id)

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