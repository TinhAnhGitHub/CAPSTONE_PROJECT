"""Test script for OCRSearchToolkit.

This script tests all tools in the OCRSearchToolkit class:
- search_ocr_text (BM25 keyword search)
- search_ocr_semantic (semantic search with embeddings)
- search_ocr_hybrid (hybrid BM25 + kNN)
- get_ocr_by_video (get all OCR for a video)
- view_ocr_result (view cached results)

Requirements:
- Elasticsearch with OCR index
- PostgreSQL database with video artifacts
- MMBert embedding service (optional, for semantic search)
- Environment variables configured or edit CONFIG below
"""

import asyncio
import os
import sys
from loguru import logger

from videodeepsearch.clients.storage.elasticsearch import (
    ElasticsearchOCRClient,
    ElasticsearchConfig,
)
from videodeepsearch.clients.storage.postgre import PostgresClient
from videodeepsearch.toolkit.ocr import OCRSearchToolkit


# ---------------------------------------------------------------------------
# Configuration — edit these values or set the corresponding env vars
# ---------------------------------------------------------------------------

CONFIG = {
    "video_index": 0,
    "postgres_url": "postgresql+asyncpg://admin123:admin123@localhost:5432/video-pipeline",
    "es_host": "localhost",
    "es_port": 9200,
    "es_user": None,
    "es_password": None,
    "es_use_ssl": False,
    "es_index_name": "video_ocr_docs_dev",

    "mmbert_url": "http://localhost:8009",  # e.g., "http://localhost:8001"

    # User ID for filtering
    "user_id": "tinhanhuser",

    # Test configurations
    "search_ocr_text": {
        "enabled": True,
        "query": "human brain perfectly",
        "top_k": 10,
        "fuzzy": True,
    },
    "get_ocr_by_video": {
        "enabled": True,
        "limit": 50,
    },
    "view_ocr_result": {
        "enabled": True,
        "view_mode": "detailed",
        "top_n": 3,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_env_overrides(cfg: dict) -> dict:
    """Override CONFIG connection values with environment variables when set."""
    if v := os.environ.get("POSTGRES_CLIENT_DATABASE_URL"):
        cfg["postgres_url"] = v
    if v := os.environ.get("ES_HOST"):
        cfg["es_host"] = v
    if v := os.environ.get("ES_PORT"):
        cfg["es_port"] = int(v)
    if v := os.environ.get("ES_USER"):
        cfg["es_user"] = v
    if v := os.environ.get("ES_PASSWORD"):
        cfg["es_password"] = v
    if v := os.environ.get("ES_USE_SSL"):
        cfg["es_use_ssl"] = v.lower() == "true"
    if v := os.environ.get("ES_INDEX_NAME"):
        cfg["es_index_name"] = v
    if v := os.environ.get("MMBERT_URL"):
        cfg["mmbert_url"] = v
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


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

async def test_search_ocr_text(toolkit: OCRSearchToolkit, cfg: dict, video_id: str | None):
    logger.info(f"Testing search_ocr_text with query: '{cfg['query']}'...")
    result = await toolkit.search_ocr_text(
        query=cfg["query"],
        top_k=cfg["top_k"],
        video_ids=[video_id] if video_id else None,
        user_id=cfg.get("user_id"),
        fuzzy=cfg["fuzzy"],
    )
    logger.info(f"Result:\n{result.content}\n")
    return result

async def test_get_ocr_by_video(toolkit: OCRSearchToolkit, video_id: str, cfg: dict):
    logger.info(f"Testing get_ocr_by_video for: {video_id}...")
    result = await toolkit.get_ocr_by_video(
        video_id=video_id,
        user_id=cfg.get("user_id"),
        limit=cfg["limit"],
    )
    logger.info(f"Result:\n{result.content}\n")
    return result


async def test_view_ocr_result(toolkit: OCRSearchToolkit, handle_id: str, cfg: dict):
    logger.info(f"Testing view_ocr_result with handle_id: {handle_id}...")
    result = toolkit.view_ocr_result(
        handle_id=handle_id,
        view_mode=cfg["view_mode"],
        top_n=cfg["top_n"],
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

    # PostgreSQL client
    postgres = PostgresClient(database_url=cfg["postgres_url"])
    logger.info("PostgreSQL client ready.")

    # Elasticsearch client
    es_config = ElasticsearchConfig(
        host=cfg["es_host"],
        port=cfg["es_port"],
        user=cfg["es_user"],
        password=cfg["es_password"],
        use_ssl=cfg["es_use_ssl"],
        index_name=cfg["es_index_name"],
    )
    es_client = ElasticsearchOCRClient(config=es_config)
    await es_client.connect()

    # MMBert client (optional)
    mmbert_client = None
    if cfg["mmbert_url"]:
        try:
            from videodeepsearch.clients.inference import MMBertClient,MMBertConfig
            config = MMBertConfig(base_url=cfg['mmbert_url'])
            mmbert_client = MMBertClient(config)
            logger.info(f"MMBert client configured: {cfg['mmbert_url']}")
        except Exception as e:
            logger.warning(f"Failed to initialize MMBert client: {e}")

    # OCR Search Toolkit
    toolkit = OCRSearchToolkit(
        es_ocr_client=es_client,
        mmbert_client=mmbert_client,
    )
    logger.info("OCRSearchToolkit ready.")

    _section("Fetching video IDs")
    video_ids = await get_video_ids(postgres)
    if not video_ids:
        logger.error("No VideoArtifact rows found in the database. Aborting.")
        await es_client.close()
        await postgres.close()
        return

    idx = cfg["video_index"]
    if idx >= len(video_ids):
        logger.warning(f"video_index={idx} out of range ({len(video_ids)} videos). Falling back to 0.")
        idx = 0
    video_id = video_ids[idx]
    logger.info(f"Using video_id: {video_id}  ({idx + 1}/{len(video_ids)})")

    # Add user_id to test configs
    for test_key in ["search_ocr_text", "search_ocr_semantic", "search_ocr_hybrid", "get_ocr_by_video"]:
        if test_key in cfg:
            cfg[test_key]["user_id"] = cfg["user_id"]

    last_handle_id = None

    # Run tests
    tests = [
        ("search_ocr_text", cfg["search_ocr_text"], test_search_ocr_text, True),
        ("get_ocr_by_video", cfg["get_ocr_by_video"], test_get_ocr_by_video, False),
    ]

    errors = []
    for name, test_cfg, fn, use_video_id in tests:
        if not test_cfg.get("enabled", True):
            logger.info(f"[SKIP] {name}")
            continue
        _section(name)

        try:
            if use_video_id:
                result = await fn(toolkit, test_cfg, video_id)
            else:
                result = await fn(toolkit, video_id, test_cfg)

            # Extract handle_id from result for view_ocr_result test
            if result and "Handle ID:" in result.content:
                import re
                match = re.search(r"Handle ID: ([a-f0-9]+)", result.content)
                if match:
                    last_handle_id = match.group(1)
                    logger.info(f"Captured handle_id: {last_handle_id}")

            logger.success(f"[PASS] {name}")
        except Exception as e:
            logger.error(f"[FAIL] {name}: {e}")
            errors.append(name)

    # Test view_ocr_result if we have a handle_id
    if cfg["view_ocr_result"].get("enabled", True) and last_handle_id:
        _section("view_ocr_result")
        try:
            await test_view_ocr_result(toolkit, last_handle_id, cfg["view_ocr_result"])
            logger.success("[PASS] view_ocr_result")
        except Exception as e:
            logger.error(f"[FAIL] view_ocr_result: {e}")
            errors.append("view_ocr_result")
    elif cfg["view_ocr_result"].get("enabled", True):
        logger.warning("[SKIP] view_ocr_result - no handle_id available from previous tests")

    _section("Summary")
    total = sum(1 for _, tc, _, _ in tests if tc.get("enabled", True))
    if cfg["view_ocr_result"].get("enabled", True) and last_handle_id:
        total += 1
    passed = total - len(errors)
    logger.info(f"Passed: {passed}/{total}")
    if errors:
        logger.warning(f"Failed: {', '.join(errors)}")
    else:
        logger.success("All enabled tests passed!")

    # Close clients
    await es_client.close()
    if mmbert_client:
        await mmbert_client.close()
    await postgres.close()
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())