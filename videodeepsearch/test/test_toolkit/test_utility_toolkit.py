"""Test script for UtilityToolkit.

This script tests all tools in the UtilityToolkit class:
- get_related_asr_from_segment
- get_related_asr_from_image
- get_adjacent_segments
- get_adjacent_images
- extract_frames_by_time_window
- extract_frame_at_time

Requirements:
- PostgreSQL database with indexed video artifacts
- MinIO storage with ASR data and videos
- Environment variables configured or edit CONFIG below
"""

import asyncio
import os
import sys
from loguru import logger

from videodeepsearch.clients.storage.postgre import PostgresClient
from videodeepsearch.clients.storage.minio import MinioStorageClient
from videodeepsearch.toolkit.utility import UtilityToolkit


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

    
    "asr_from_segment": {
        "enabled": True,
        "segment_start": "00:00:10.000",
        "segment_end": "00:01:20.000",
        "window_seconds": 10.0,
    },
    "asr_from_image": {
        "enabled": True,
        "image_timestamp": "00:01:20.000",
        "window_seconds": 20.0,
    },
    "adjacent_segments": {
        "enabled": True,
        "pivot_start_frame": 2739,
        "pivot_end_frame": 2739,
        "hop": 2,
        "include_range": True,
        'direction': 'forward',  # or 'backward'    
    },
    "adjacent_images": {
        "enabled": True,
        "image_frame_index": 10,
        "hop": 3,
        "include_range": True,
    },
    "extract_frames_by_window": {
        "enabled": True,
        "start_time": "00:00:05.000",
        "end_time": "00:00:07.000",
        "fps_sample_rate": 1,
    },
    "extract_frame_at_time": {
        "enabled": True,
        "timestamp": "00:00:10.000",
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
        
    
    return cfg


def _section(title: str) -> None:
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# DB helper
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




async def test_get_related_asr_from_segment(toolkit: UtilityToolkit, video_id: str, cfg: dict):
    logger.info("Testing get_related_asr_from_segment...")
    result = await toolkit.get_related_asr_from_segment(
        video_id=video_id,
        segment_start_time=cfg["segment_start"],
        segment_end_time=cfg["segment_end"],
        window_seconds=cfg["window_seconds"],
    )
    logger.info(f"Result:\n{result.content}\n")
    return result


async def test_get_related_asr_from_image(toolkit: UtilityToolkit, video_id: str, cfg: dict):
    logger.info("Testing get_related_asr_from_image...")
    result = await toolkit.get_related_asr_from_image(
        video_id=video_id,
        image_timestamp=cfg["image_timestamp"],
        window_seconds=cfg["window_seconds"],
    )
    logger.info(f"Result:\n{result.content}\n")
    return result


async def test_get_adjacent_segments(toolkit: UtilityToolkit, video_id: str, cfg: dict):
    logger.info("Testing get_adjacent_segments...")
    result_forward = await toolkit.get_adjacent_segments(
        video_id=video_id,
        pivot_start_frame=cfg["pivot_start_frame"],
        pivot_end_frame=cfg["pivot_end_frame"],
        hop=cfg["hop"],
        direction="forward",
        include_range=cfg["include_range"],
    )
    logger.info(f"Forward result:\n{result_forward.content}\n")

    result_backward = await toolkit.get_adjacent_segments(
        video_id=video_id,
        pivot_start_frame=cfg["pivot_start_frame"],
        pivot_end_frame=cfg["pivot_end_frame"],
        hop=cfg["hop"],
        direction="backward",
        include_range=cfg["include_range"],
    )
    logger.info(f"Backward result:\n{result_backward.content}\n")
    return result_forward, result_backward


async def test_get_adjacent_images(toolkit: UtilityToolkit, video_id: str, cfg: dict):
    logger.info("Testing get_adjacent_images...")
    result_forward = await toolkit.get_adjacent_images(
        video_id=video_id,
        image_frame_index=cfg["image_frame_index"],
        hop=cfg["hop"],
        direction="forward",
        include_range=cfg["include_range"],
    )
    logger.info(f"Forward result:\n{result_forward.content}\n")

    result_backward = await toolkit.get_adjacent_images(
        video_id=video_id,
        image_frame_index=cfg["image_frame_index"],
        hop=cfg["hop"],
        direction="backward",
        include_range=cfg["include_range"],
    )
    logger.info(f"Backward result:\n{result_backward.content}\n")
    return result_forward, result_backward


async def test_extract_frames_by_time_window(toolkit: UtilityToolkit, video_id: str, cfg: dict):
    logger.info("Testing extract_frames_by_time_window...")
    result = await toolkit.extract_frames_by_time_window(
        video_id=video_id,
        start_time=cfg["start_time"],
        end_time=cfg["end_time"],
        fps_sample_rate=cfg["fps_sample_rate"],
    )
    logger.info(f"Result: {result.content}")
    if result.images:
        logger.info(f"Extracted {len(result.images)} frames")
        
    
    return result


async def test_extract_frame_at_time(toolkit: UtilityToolkit, video_id: str, cfg: dict):
    logger.info("Testing extract_frame_at_time...")
    result = await toolkit.extract_frame_at_time(
        video_id=video_id,
        timestamp=cfg["timestamp"],
    )
    logger.info(f"Result: {result.content}")
    if result.images:
        logger.info("Successfully extracted single frame")
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
    toolkit = UtilityToolkit(postgres_client=postgres, minio_client=minio)
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

    tests = [
        ("get_related_asr_from_segment",  cfg["asr_from_segment"],         test_get_related_asr_from_segment),
        ("get_related_asr_from_image",    cfg["asr_from_image"],            test_get_related_asr_from_image),
        ("get_adjacent_segments",         cfg["adjacent_segments"],         test_get_adjacent_segments),
        ("get_adjacent_images",           cfg["adjacent_images"],           test_get_adjacent_images),
        ("extract_frames_by_time_window", cfg["extract_frames_by_window"],  test_extract_frames_by_time_window),
        ("extract_frame_at_time",         cfg["extract_frame_at_time"],     test_extract_frame_at_time),
    ]

    errors = []
    for name, test_cfg, fn in tests:
        if not test_cfg.get("enabled", True):
            logger.info(f"[SKIP] {name}")
            continue
        _section(name)
    
        await fn(toolkit, video_id, test_cfg)
        logger.success(f"[PASS] {name}")
        

    _section("Summary")
    total = sum(1 for _, tc, _ in tests if tc.get("enabled", True))
    logger.info(f"Passed: {total - len(errors)}/{total}")
    if errors:
        logger.warning(f"Failed: {', '.join(errors)}")
    else:
        logger.success("All enabled tests passed!")

    await postgres.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())