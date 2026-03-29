"""Test script for LLMToolkit.

This script tests all tools in the LLMToolkit class:
- enhance_visual_query: Generate CLIP-optimized visual query variations
- enhance_textual_query: Generate semantic query variations for caption/event search

Requirements:
- OpenRouter API key configured
- Environment variables configured or edit CONFIG below
"""

import asyncio
import os
import sys
from loguru import logger

from agno.models.openrouter import OpenRouter
from videodeepsearch.toolkit.llm import LLMToolkit
from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Configuration — edit these values or set the corresponding env vars
# ---------------------------------------------------------------------------

CONFIG = {
    # OpenRouter configuration
    "api_key": os.getenv(key='OPENROUTER_API_KEY'),# Set via OPENROUTER_API_KEY env var 
    "base_url": "https://openrouter.ai/api/v1",
    "model_id": "qwen/qwen3.5-35b-a3b", 

    # Test queries
    "visual_query_test": {
        "enabled": True,
        "raw_query": "a person walking in a park",
        "variants": [
            "close-up shot focusing on the person's movement",
            "wide angle showing the park landscape",
            "golden hour lighting with warm colors",
        ],
    },
    "textual_query_test": {
        "enabled": True,
        "raw_query": "someone giving a presentation in a meeting room",
        "variants": [
            "focus on the speaker's gestures and expressions",
            "audience perspective showing engagement",
            "technical content visible on screen",
        ],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_env_overrides(cfg: dict) -> dict:
    """Override CONFIG values with environment variables when set."""
    if v := os.environ.get("OPENROUTER_API_KEY"):
        cfg["api_key"] = v
    if v := os.environ.get("OPENROUTER_BASE_URL"):
        cfg["base_url"] = v
    if v := os.environ.get("OPENROUTER_MODEL_ID"):
        cfg["model_id"] = v
    return cfg


def _section(title: str) -> None:
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

async def test_enhance_visual_query(toolkit: LLMToolkit, cfg: dict):
    """Test enhance_visual_query tool."""
    logger.info("Testing enhance_visual_query...")
    logger.info(f"Raw query: {cfg['raw_query']}")
    logger.info(f"Variants: {cfg['variants']}")

    result = await toolkit.enhance_visual_query.entrypoint(
        toolkit,
        raw_query=cfg["raw_query"],
        variants=cfg["variants"],
    )

    logger.info(f"Result:\n{result.content}\n")

    # Check that we got a valid result (not an error)
    if result.content.startswith("Error"):
        raise Exception(f"Tool returned error: {result.content}")

    # Check that we got multiple enhanced queries
    lines = [l for l in result.content.split("\n") if l.strip() and not l.strip().startswith("Generated")]
    if len(lines) < 1:
        raise Exception("No enhanced queries returned")

    logger.success(f"Generated {len(lines)} visual query variations")
    return result


async def test_enhance_textual_query(toolkit: LLMToolkit, cfg: dict):
    """Test enhance_textual_query tool."""
    logger.info("Testing enhance_textual_query...")
    logger.info(f"Raw query: {cfg['raw_query']}")
    logger.info(f"Variants: {cfg['variants']}")

    result = await toolkit.enhance_textual_query.entrypoint(
        toolkit,
        raw_query=cfg["raw_query"],
        variants=cfg["variants"],
    )

    logger.info(f"Result:\n{result.content}\n")

    # Check that we got a valid result (not an error)
    if result.content.startswith("Error"):
        raise Exception(f"Tool returned error: {result.content}")

    # Check that we got multiple enhanced queries
    lines = [l for l in result.content.split("\n") if l.strip() and not l.strip().startswith("Generated")]
    if len(lines) < 1:
        raise Exception("No enhanced queries returned")

    logger.success(f"Generated {len(lines)} textual query variations")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    cfg = _apply_env_overrides(CONFIG)

    if not cfg["api_key"]:
        logger.error("No OpenRouter API key found. Set OPENROUTER_API_KEY or edit CONFIG.")
        sys.exit(1)

    _section("Initializing LLM model and toolkit")

    model = OpenRouter(
        id=cfg["model_id"],
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
    )

    toolkit = LLMToolkit(llm_client=model)
    logger.info(f"Model: {cfg['model_id']}")
    logger.info("Toolkit ready.")

    tests = [
        ("enhance_visual_query", cfg["visual_query_test"], test_enhance_visual_query),
        ("enhance_textual_query", cfg["textual_query_test"], test_enhance_textual_query),
    ]

    errors = []
    for name, test_cfg, fn in tests:
        if not test_cfg.get("enabled", True):
            logger.info(f"[SKIP] {name}")
            continue

        _section(name)
        try:
            await fn(toolkit, test_cfg)
            logger.success(f"[PASS] {name}")
        except Exception as e:
            logger.error(f"[FAIL] {name}: {e}")
            errors.append(name)

    _section("Summary")
    total = sum(1 for _, tc, _ in tests if tc.get("enabled", True))
    passed = total - len(errors)
    logger.info(f"Passed: {passed}/{total}")

    if errors:
        logger.warning(f"Failed: {', '.join(errors)}")
        sys.exit(1)
    else:
        logger.success("All enabled tests passed!")

    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())