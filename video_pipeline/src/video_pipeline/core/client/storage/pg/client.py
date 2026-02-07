from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool

from .config import PgConfig
from .schema import ArtifactLineageSchema, ArtifactMetadata, ArtifactSchema


class PostgresClient:
    def __init__(self, config: PgConfig):
        self.engine = create_async_engine(
            url=config.database_url,
            echo=False,
            pool_pre_ping=True,
            poolclass=AsyncAdaptedQueuePool,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout,
            pool_recycle=config.pool_recycle,
        )

        self._sessionmaker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    def get_session(self) -> AsyncSession:
        return self._sessionmaker()

    async def save_artifact(self, metadata: ArtifactMetadata) -> str:
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
