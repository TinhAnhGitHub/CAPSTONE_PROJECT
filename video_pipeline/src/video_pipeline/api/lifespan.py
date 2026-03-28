from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from loguru import logger
from prefect.client.orchestration import get_client
from prefect.exceptions import ObjectNotFound
from dotenv import load_dotenv

FLOW_NAME = "Single Video Processing Flow"
DEPLOYMENT_NAME = os.getenv("PREFECT_DEPLOYMENT_NAME", "poc-deployment")
DEPLOY_IDENTIFIER = f"{FLOW_NAME}/{DEPLOYMENT_NAME}"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Verify the Prefect deployment is registered before accepting traffic."""
    try:
        async with get_client() as client:
            deployment = await client.read_deployment_by_name(DEPLOY_IDENTIFIER)
            logger.info(
                f"Prefect deployment found: '{DEPLOY_IDENTIFIER}' (id={deployment.id})"
            )
    except ObjectNotFound:
        logger.warning(
            f"Prefect deployment '{DEPLOY_IDENTIFIER}' not found. "
            "Requests will fail until 'prefect deploy --name poc-deployment' is run inside the worker."
        )
    except Exception as exc:
        logger.warning(f"Could not verify Prefect deployment at startup: {exc}")

    yield
