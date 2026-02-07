from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool

from .config import PgConfig


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
