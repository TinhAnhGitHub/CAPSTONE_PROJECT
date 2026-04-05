"""Test script for VideoMetadataToolkit.

This script tests all tools in the VideoMetadataToolkit class:
- list_user_videos
- get_video_metadata
- get_video_summary
- get_video_timeline (segment, shot, minute granularity)

Requirements:
- PostgreSQL database with indexed video artifacts
- MinIO storage with video data
- Environment variables configured or edit CONFIG below
"""

import asyncio
import os
import sys
from loguru import logger

from videodeepsearch.clients.storage.postgre import PostgresClient
from videodeepsearch.clients.storage.minio import MinioStorageClient
from videodeepsearch.toolkit.video_metadata import VideoMetadataToolkit


# ---------------------------------------------------------------------------
# Configuration — edit these values or set the corresponding env vars
# ---------------------------------------------------------------------------

CONFIG = {
    # Which video index to use from the list fetched from the DB (0 = first)
    "video_index": 0,

    # Connection
    "postgres_url": "postgresql+asyncpg://admin123:admin123@localhost:5432/video-pipeline",
    "minio_host": "localhost",
    "minio_port": "9000",
    "minio_access_key": "minioadmin",
    "minio_secret_key": "minioadmin",
    "minio_secure": False,

    # User ID for list_user_videos test
    "user_id": "tinhanhuser",

    # Test configurations
    "list_user_videos": {
        "enabled": True,
        "limit": 10,
        "offset": 0,
    },
    "get_video_metadata": {
        "enabled": True,
    },
    "get_video_timeline_segment": {
        "enabled": True,
        "granularity": "segment",
    },
    "get_video_timeline_shot": {
        "enabled": True,
        "granularity": "shot",
    },
    "get_video_timeline_minute": {
        "enabled": True,
        "granularity": "minute",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_env_overrides(cfg: dict) -> dict:
    """Override CONFIG connection values with environment variables when set."""
    if v := os.environ.get("POSTGRES_CLIENT_DATABASE_URL"):
        cfg["postgres_url"] = v
    if v := os.environ.get("MINIO_STORAGE_CLIENT_HOST"):
        cfg["minio_host"] = v
    if v := os.environ.get("MINIO_STORAGE_CLIENT_PORT"):
        cfg["minio_port"] = v
    if v := os.environ.get("MINIO_STORAGE_CLIENT_ACCESS_KEY"):
        cfg["minio_access_key"] = v
    if v := os.environ.get("MINIO_STORAGE_CLIENT_SECRET_KEY"):
        cfg["minio_secret_key"] = v
    if v := os.environ.get("MINIO_STORAGE_CLIENT_SECURE"):
        cfg["minio_secure"] = v.lower() == "true"
    if v := os.environ.get("VIDEO_METADATA_TEST_USER_ID"):
        cfg["user_id"] = v

    return cfg


def _section(title: str) -> None:
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def get_video_ids(postgres: PostgresClient) -> list[str]:
    from sqlalchemy import select
    from videodeepsearch.clients.storage.postgre.schema import ArtifactSchema

    async with postgres.get_session() as session:
        result = await session.execute(
            select(ArtifactSchema.artifact_id).where(
                ArtifactSchema.artifact_type == "VideoArtifact"
            )
        )
        return [row[0] for row in result.fetchall()]


async def get_user_id_from_video(postgres: PostgresClient, video_id: str) -> str | None:
    from sqlalchemy import select
    from videodeepsearch.clients.storage.postgre.schema import ArtifactSchema

    async with postgres.get_session() as session:
        result = await session.execute(
            select(ArtifactSchema.user_id).where(
                ArtifactSchema.artifact_id == video_id
            )
        )
        row = result.fetchone()
        return row[0] if row else None


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

async def test_list_user_videos(toolkit: VideoMetadataToolkit, cfg: dict):
    logger.info("Testing list_user_videos...")
    result = await toolkit.list_user_videos(
        user_id=cfg["user_id"],
        limit=cfg["limit"],
        offset=cfg["offset"],
    )
    logger.info(f"Result:\n{result.content}\n")
    return result


async def test_get_video_metadata(toolkit: VideoMetadataToolkit, video_id: str, cfg: dict):
    logger.info("Testing get_video_metadata...")
    result = await toolkit.get_video_metadata(video_id=video_id)
    logger.info(f"Result:\n{result.content}\n")
    return result




async def test_get_video_timeline(toolkit: VideoMetadataToolkit, video_id: str, cfg: dict):
    granularity = cfg["granularity"]
    logger.info(f"Testing get_video_timeline (granularity={granularity})...")
    result = await toolkit.get_video_timeline(
        video_id=video_id,
        granularity=granularity,
    )
    logger.info(f"Result:\n{result.content}\n")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    cfg = _apply_env_overrides(CONFIG)

    if not cfg["postgres_url"]:
        logger.error("No Postgres URL found. Set POSTGRES_CLIENT_DATABASE_URL or edit CONFIG.")
        sys.exit(1)

    _section("Initializing clients")
    postgres = PostgresClient(database_url=cfg["postgres_url"])
    minio = MinioStorageClient(
        host=cfg["minio_host"],
        port=cfg["minio_port"],
        access_key=cfg["minio_access_key"],
        secret_key=cfg["minio_secret_key"],
        secure=cfg["minio_secure"],
    )
    toolkit = VideoMetadataToolkit(postgres_client=postgres, minio_client=minio)
    logger.info("Clients ready.")

    _section("Fetching video IDs")
    video_ids = await get_video_ids(postgres)
    if not video_ids:
        logger.error("No VideoArtifact rows found in the database. Aborting.")
        await postgres.close()
        return

    idx = cfg["video_index"]
    if idx >= len(video_ids):
        logger.warning(f"video_index={idx} out of range ({len(video_ids)} videos). Falling back to 0.")
        idx = 0
    video_id = video_ids[idx]
    logger.info(f"Using video_id: {video_id}  ({idx + 1}/{len(video_ids)})")

    # Get user_id from video if not set
    if not cfg.get("user_id"):
        user_id = await get_user_id_from_video(postgres, video_id)
        if user_id:
            cfg["user_id"] = user_id
            cfg["list_user_videos"]["user_id"] = user_id
            logger.info(f"Using user_id from video: {user_id}")

    # Update list_user_videos config with user_id
    cfg["list_user_videos"]["user_id"] = cfg["user_id"]

    tests = [
        ("list_user_videos", cfg["list_user_videos"], test_list_user_videos, False),
        ("get_video_metadata", cfg["get_video_metadata"], test_get_video_metadata, True),
        ("get_video_timeline_segment", cfg["get_video_timeline_segment"], test_get_video_timeline, True),
        ("get_video_timeline_shot", cfg["get_video_timeline_shot"], test_get_video_timeline, True),
        ("get_video_timeline_minute", cfg["get_video_timeline_minute"], test_get_video_timeline, True),
    ]

    errors = []
    for name, test_cfg, fn, needs_video_id in tests:
        if not test_cfg.get("enabled", True):
            logger.info(f"[SKIP] {name}")
            continue
        _section(name)

        
        if needs_video_id:
            await fn(toolkit, video_id, test_cfg) #type:ignore
        else:
            await fn(toolkit, test_cfg) #type:ignore
        logger.success(f"[PASS] {name}")
       

    _section("Summary")
    total = sum(1 for _, tc, _, _ in tests if tc.get("enabled", True))
    logger.info(f"Passed: {total - len(errors)}/{total}")
    if errors:
        logger.warning(f"Failed: {', '.join(errors)}")
    else:
        logger.success("All enabled tests passed!")

    await postgres.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())