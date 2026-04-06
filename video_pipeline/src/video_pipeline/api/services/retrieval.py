"""Video retrieval service to fetch all data by video_id."""

from __future__ import annotations

import asyncio
from typing import Any

from arango.client import ArangoClient
from elasticsearch import AsyncElasticsearch
from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from video_pipeline.config import get_settings
from video_pipeline.core.client.storage.pg.schema import ArtifactSchema, ArtifactLineageSchema


class VideoRetrievalService:
    """Service to retrieve all data associated with a video_id."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def _get_postgres_session(self) -> AsyncSession:
        """Get a PostgreSQL session."""
        engine = create_async_engine(
            url=self.settings.postgres.connection_string,
            echo=False,
            poolclass=NullPool,
        )
        sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        return sessionmaker()

    def _get_arango_db(self):
        """Get ArangoDB database connection."""
        client = ArangoClient(hosts=self.settings.arango.host)
        return client.db(self.settings.arango.database)

    def _get_qdrant_client(self) -> AsyncQdrantClient:
        """Get a Qdrant client."""
        return AsyncQdrantClient(
            host=self.settings.qdrant.host,
            port=self.settings.qdrant.port,
        )

    def _get_elasticsearch_client(self) -> AsyncElasticsearch:
        """Get an Elasticsearch client."""
        es = self.settings.elasticsearch
        url = f"http://{es.host}:{es.port}"
        return AsyncElasticsearch(url)

    async def _fetch_postgres(self, video_id: str, session: AsyncSession) -> dict[str, Any]:
        """Fetch artifacts and lineage from PostgreSQL."""
        result = {"artifacts": [], "lineage": []}

        stmt = select(ArtifactSchema).where(
            ArtifactSchema.artifact_metadata["related_video_id"].as_string() == video_id
        )
        db_result = await session.execute(stmt)
        artifacts = db_result.scalars().all()

        stmt2 = select(ArtifactSchema).where(ArtifactSchema.artifact_id == video_id)
        db_result2 = await session.execute(stmt2)
        video_artifact = db_result2.scalars().first()

        if video_artifact and video_artifact not in artifacts:
            artifacts = list(artifacts) + [video_artifact]

        for a in artifacts:
            result["artifacts"].append({
                "artifact_id": a.artifact_id,
                "artifact_type": a.artifact_type,
                "minio_url": a.minio_url,
                "user_id": a.user_id,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "metadata": a.artifact_metadata,
            })

        # Get lineage relationships
        artifact_ids = [a.artifact_id for a in artifacts]
        if artifact_ids:
            lineage_stmt = select(ArtifactLineageSchema).where(
                ArtifactLineageSchema.child_artifact_id.in_(artifact_ids)
            )
            lineage_result = await session.execute(lineage_stmt)
            lineages = lineage_result.scalars().all()

            for l in lineages:
                result["lineage"].append({
                    "parent_artifact_id": l.parent_artifact_id,
                    "child_artifact_id": l.child_artifact_id,
                })

        return result

    def _fetch_arango(self, video_id: str) -> dict[str, Any]:
        """Fetch KG data from ArangoDB."""
        result = {
            "entities": [],
            "events": [],
            "micro_events": [],
            "communities": [],
            "relationships": [],
            "event_edges": [],
        }

        try:
            db = self._get_arango_db()

            # Fetch entities
            query = "FOR doc IN entities FILTER doc.video_id == @video_id RETURN doc"
            cursor = db.aql.execute(query, bind_vars={"video_id": video_id})
            result["entities"] = list(cursor) #type:ignore

            # Fetch events
            query = "FOR doc IN events FILTER doc.video_id == @video_id RETURN doc"
            cursor = db.aql.execute(query, bind_vars={"video_id": video_id})
            result["events"] = list(cursor) #type:ignore

            # Fetch micro_events
            query = "FOR doc IN micro_events FILTER doc.video_id == @video_id RETURN doc"
            cursor = db.aql.execute(query, bind_vars={"video_id": video_id})
            result["micro_events"] =    list(cursor) #type:ignore

            # Fetch communities
            query = "FOR doc IN communities FILTER doc.video_id == @video_id RETURN doc"
            cursor = db.aql.execute(query, bind_vars={"video_id": video_id})
            result["communities"] = list(cursor) #type:ignore

            # Fetch entity relationships
            query = "FOR doc IN entity_relations FILTER doc.video_id == @video_id RETURN doc"
            cursor = db.aql.execute(query, bind_vars={"video_id": video_id})
            result["relationships"] = list(cursor) #type:ignore

            # Fetch event sequences
            query = "FOR doc IN event_sequences FILTER doc.video_id == @video_id RETURN doc"
            cursor = db.aql.execute(query, bind_vars={"video_id": video_id})
            result["event_edges"] = list(cursor) #type:ignore

        except Exception as e:
            logger.warning(f"[ArangoDB] Error fetching data: {e}")
            result["error"] = str(e) #type:ignore

        return result

    async def _fetch_qdrant(
        self,
        video_id: str,
        include_vectors: bool = False,
    ) -> dict[str, Any]:
        """Fetch embeddings from Qdrant."""
        result = {}
        client = self._get_qdrant_client()
        collection_base = self.settings.qdrant.collection_base

        collections = [
            f"{collection_base}_image",
            f"{collection_base}_caption",
            f"{collection_base}_segment",
            f"{collection_base}_segment_caption",
            f"{collection_base}_audio_transcript",
        ]

        try:
            for collection_name in collections:
                try:
                    exists = await client.collection_exists(collection_name)
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

                    # Scroll through all matching points
                    points, _ = await client.scroll(
                        collection_name=collection_name,
                        scroll_filter=filter_condition,
                        limit=1000,
                        with_vectors=include_vectors,
                    )

                    result[collection_name] = {
                        "count": len(points),
                        "ids": [str(p.id) for p in points],
                        "payloads": [
                            {"id": str(p.id), **p.payload} #type:ignore
                            for p in points[:10]  # Limit payload preview
                        ],
                    }

                except Exception as e:
                    result[collection_name] = {"error": str(e)}

        finally:
            await client.close()

        return result

    async def _fetch_elasticsearch(self, video_id: str) -> dict[str, Any]:
        """Fetch OCR documents from Elasticsearch."""
        result = {"count": 0, "documents": []}
        client = self._get_elasticsearch_client()

        try:
            resp = await client.search(
                index=self.settings.elasticsearch.index_name,
                query={"term": {"video_id": video_id}},
                size=100,
            )

            hits = resp.get("hits", {}).get("hits", [])
            result["count"] = len(hits)
            result["documents"] = [
                {"id": hit["_id"], **hit["_source"]}
                for hit in hits[:10]  # Limit preview
            ]

        except Exception as e:
            logger.warning(f"[Elasticsearch] Error fetching data: {e}")
            result["error"] = str(e) #type:ignore

        finally:
            await client.close()

        return result

    async def get_video_data(
        self,
        video_id: str,
        sources: list[str] | None = None,
        include_vectors: bool = False,
    ) -> dict[str, Any]:
        """Retrieve all data associated with a video_id.

        Args:
            video_id: The video ID to retrieve data for
            sources: Optional list of sources to fetch from.
                     Options: ["postgres", "arango", "qdrant", "elasticsearch"]
                     If None, fetches from all sources.
            include_vectors: Whether to include embedding vectors in Qdrant results

        Returns:
            Dict with data from each requested source
        """
        logger.info(f"[VideoRetrieval] Fetching data for video_id={video_id}")

        results: dict[str, Any] = {"video_id": video_id}
        all_sources = ["postgres", "arango", "qdrant", "elasticsearch"]
        sources_to_fetch = sources or all_sources

        session = None
        tasks = []

        try:
            # Prepare async tasks
            if "postgres" in sources_to_fetch:
                session = await self._get_postgres_session()

            async def fetch_postgres():
                if session:
                    return await self._fetch_postgres(video_id, session)
                return {}

            async def fetch_arango():
                return await asyncio.to_thread(self._fetch_arango, video_id)

            async def fetch_qdrant():
                return await self._fetch_qdrant(video_id, include_vectors)

            async def fetch_elasticsearch():
                return await self._fetch_elasticsearch(video_id)

            # Build task list
            task_map = {
                "postgres": ("postgres", fetch_postgres),
                "arango": ("arango", fetch_arango),
                "qdrant": ("qdrant", fetch_qdrant),
                "elasticsearch": ("elasticsearch", fetch_elasticsearch),
            }

            for source in sources_to_fetch:
                if source in task_map:
                    key, coro = task_map[source]
                    tasks.append((key, coro()))

            # Execute all fetches in parallel
            for key, task in tasks:
                try:
                    results[key] = await task
                except Exception as e:
                    logger.warning(f"[VideoRetrieval] Error fetching {key}: {e}")
                    results[key] = {"error": str(e)}

        finally:
            if session:
                await session.close()

        logger.info(f"[VideoRetrieval] Completed for video_id={video_id}")
        return results