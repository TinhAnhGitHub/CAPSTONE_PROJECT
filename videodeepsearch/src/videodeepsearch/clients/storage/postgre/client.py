from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from videodeepsearch.clients.storage.postgre.schema import (
    ArtifactLineageSchema,
    ArtifactMetadata,
    ArtifactSchema,
    Base,
)


class PostgresClient:
    """Async PostgreSQL client for artifact metadata operations.

    Provides methods for storing and retrieving artifact metadata,
    including lineage relationships between artifacts.
    """

    def __init__(self, database_url: str):
        """Initialize the PostgreSQL client.

        Args:
            database_url: SQLAlchemy async database URL
                (e.g., "postgresql+asyncpg://user:pass@host/db")
        """
        self.database_url = database_url
        self.engine = create_async_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
            poolclass=NullPool,
        )

    def get_session(self) -> AsyncSession:
        """Create a new async session.

        Returns:
            New AsyncSession instance
        """
        sessionmaker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        return sessionmaker()

    async def initialize(self) -> None:
        """Create all tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Close the database engine."""
        await self.engine.dispose()

    async def get_artifact(self, artifact_id: str) -> ArtifactMetadata | None:
        """Retrieve artifact metadata by ID.

        Args:
            artifact_id: Unique artifact identifier

        Returns:
            ArtifactMetadata if found, None otherwise
        """
        async with self.get_session() as session:
            result = await session.get(ArtifactSchema, artifact_id)
            if not result:
                return None

            return ArtifactMetadata(
                artifact_id=result.artifact_id,
                artifact_type=result.artifact_type,
                minio_url=result.minio_url,
                created_at=result.created_at,
                user_id=result.user_id,
                artifact_metadata=result.artifact_metadata or {},
            )

    async def get_children_artifact(
        self,
        artifact_id: str,
        filter_artifact_type: list[str] | None = None,
    ) -> list[ArtifactMetadata]:
        """Retrieve all children of an artifact recursively.

        Args:
            artifact_id: Parent artifact ID to start from
            filter_artifact_type: Optional list of artifact types to filter by

        Returns:
            List of ArtifactMetadata for all descendant artifacts
        """
        async with self.get_session() as session:
            visited: set[str] = set()
            results: list[ArtifactMetadata] = []

            async def fetch_children_recursive(current_id: str) -> None:
                if current_id in visited:
                    return
                visited.add(current_id)

                lineage_query = select(ArtifactLineageSchema.child_artifact_id).where(
                    ArtifactLineageSchema.parent_artifact_id == current_id
                )
                lineage_result = await session.execute(lineage_query)

                child_ids = [row[0] for row in lineage_result]
                if not child_ids:
                    return

                artifact_query = select(ArtifactSchema).where(
                    ArtifactSchema.artifact_id.in_(child_ids)
                )
                artifact_result = await session.execute(artifact_query)
                artifacts = artifact_result.scalars().all()

                for a in artifacts:
                    results.append(
                        ArtifactMetadata(
                            artifact_id=a.artifact_id,
                            artifact_type=a.artifact_type,
                            minio_url=a.minio_url,
                            created_at=a.created_at,
                            user_id=a.user_id,
                            artifact_metadata=a.artifact_metadata or {},
                            lineage_parents=[current_id],
                        )
                    )

                for a in artifacts:
                    await fetch_children_recursive(a.artifact_id)

            await fetch_children_recursive(current_id=artifact_id)

            if filter_artifact_type:
                results = list(
                    filter(
                        lambda x: x.artifact_type in filter_artifact_type,
                        results,
                    )
                )
            return results

    async def get_caption_by_image_id(self, image_id: str) -> str:
        """Retrieve caption text for an image by finding the ImageCaptionArtifact.

        The ImageCaptionArtifact shares the same parent (ImageArtifact) as
        ImageEmbeddingArtifact. This method finds the caption artifact and
        extracts the caption text from its metadata.

        Args:
            image_id: The ImageArtifact ID (parent of both embedding and caption artifacts)

        Returns:
            Caption text string, or empty string if not found
        """
        async with self.get_session() as session:
            # Get children of ImageArtifact filtered by ImageCaptionArtifact type
            lineage_query = select(ArtifactLineageSchema.child_artifact_id).where(
                ArtifactLineageSchema.parent_artifact_id == image_id
            )
            lineage_result = await session.execute(lineage_query)
            child_ids = [row[0] for row in lineage_result]

            if not child_ids:
                return ""

            # Query for ImageCaptionArtifact among children
            artifact_query = select(ArtifactSchema).where(
                ArtifactSchema.artifact_id.in_(child_ids),
                ArtifactSchema.artifact_type == "ImageCaptionArtifact",
            )
            artifact_result = await session.execute(artifact_query)
            caption_artifact = artifact_result.scalar_one_or_none()

            if not caption_artifact:
                return ""

            # Extract caption from metadata
            metadata = caption_artifact.artifact_metadata or {}
            return metadata.get("caption", "")