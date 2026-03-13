import asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from .config import PgConfig
from .schema import ArtifactLineageSchema, ArtifactMetadata, ArtifactSchema, Base


class PostgresClient:
    """PostgresClient with lazy engine creation and NullPool.

    Uses NullPool to avoid cross-event-loop connection reuse issues
    when running in Dask/Prefect workers. Each task gets a fresh
    connection, eliminating "Future attached to different loop" errors.
    """

    def __init__(self, config: PgConfig):
        self._config = config
        self._engine = None
        self._sessionmaker = None

    def _ensure_engine(self):
        """Lazily create engine bound to the CURRENT event loop."""
        if self._engine is not None:
            return

        self._engine = create_async_engine(
            url=self._config.database_url,
            echo=False,
            # NullPool = no connection reuse across calls
            # Eliminates cross-loop pool issues at cost of one TCP handshake per session
            poolclass=NullPool,
        )
        self._sessionmaker = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def initialize(self):
        """Initialize database tables."""
        self._ensure_engine()
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    def get_session(self) -> AsyncSession:
        """Get a new session from the sessionmaker."""
        self._ensure_engine()
        return self._sessionmaker()

    async def save_artifact(self, metadata: ArtifactMetadata) -> str:
        """Save artifact metadata to PostgreSQL."""
        self._ensure_engine()
        async with self.get_session() as session:
            artifact = ArtifactSchema(
                artifact_id=metadata.artifact_id,
                artifact_type=metadata.artifact_type,
                minio_url=metadata.minio_url,
                created_at=metadata.created_at,
                user_id=metadata.user_id,
                artifact_metadata=metadata.artifact_metadata,
            )
            session.add(artifact)
            await session.flush()

            if len(metadata.lineage_parents) > 0:
                for parent_id in {pid for pid in metadata.lineage_parents if pid}:
                    lineage = ArtifactLineageSchema(
                        parent_artifact_id=parent_id,
                        child_artifact_id=metadata.artifact_id,
                    )
                    session.add(lineage)

            await session.commit()
            return metadata.artifact_id

    async def get_artifact(self, artifact_id: str) -> ArtifactMetadata | None:
        """Get artifact metadata from PostgreSQL."""
        self._ensure_engine()
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
                artifact_metadata=result.artifact_metadata,
            )

    async def dispose(self):
        """Dispose of the engine and connections."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None
