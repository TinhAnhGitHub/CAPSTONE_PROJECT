"""Per-task PostgresClient factory for Dask/Prefect workers.

Creates a fresh PostgresClient for each task to avoid cross-event-loop
issues. Combined with NullPool in the client, this eliminates all
"Future attached to different loop" errors.

Usage:
    from video_pipeline.core.client.storage.pg.runtime import create_postgres_client

    async with create_postgres_client() as db:
        async with db.get_session() as session:
            ...
"""

from contextlib import asynccontextmanager

from video_pipeline.config import get_settings
from video_pipeline.core.client.storage.pg.client import PostgresClient
from video_pipeline.core.client.storage.pg.config import PgConfig


@asynccontextmanager
async def create_postgres_client():
    """Create a fresh PostgresClient for the current task.

    Each call creates a new client with its own engine bound to the
    current event loop. This is safe for Dask/Prefect workers that
    run tasks across multiple threads with different event loops.

    Yields:
        PostgresClient with fresh engine for current event loop.
    """
    settings = get_settings()
    client = PostgresClient(
        config=PgConfig(database_url=settings.postgres.connection_string) #type:ignore
    )
    try:
        await client.initialize()
        yield client
    finally:
        await client.dispose()


async def get_postgres_client() -> PostgresClient:
    """Get a fresh PostgresClient for the current task.

    Note: Unlike traditional singleton patterns, this creates a NEW
    client each time. This is intentional to avoid cross-loop issues.
    The caller is responsible for calling dispose() when done.

    Returns:
        Fresh PostgresClient with engine bound to current event loop.
    """
    settings = get_settings()
    client = PostgresClient(
        config=PgConfig(database_url=settings.postgres.connection_string) #type:ignore
    )
    await client.initialize()
    return client


async def shutdown_postgres_client(client: PostgresClient) -> None:
    """Dispose of a PostgresClient created by get_postgres_client.

    Args:
        client: The client to dispose of.
    """
    await client.dispose()
