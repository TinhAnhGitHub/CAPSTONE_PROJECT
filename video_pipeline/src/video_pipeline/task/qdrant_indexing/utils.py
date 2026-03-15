"""Utility functions for Qdrant indexing tasks."""

from __future__ import annotations

import io
import re

import numpy as np
from prefect.artifacts import acreate_markdown_artifact, acreate_table_artifact
from qdrant_client.models import SparseVector

from video_pipeline.config import get_settings
from video_pipeline.core.client.inference import SpladeClient, SpladeConfig
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.qdrant.client import QdrantStorageClient
from video_pipeline.core.client.storage.qdrant.config import QdrantConfig
from video_pipeline.task.qdrant_indexing.config import ADDITIONAL_KWARGS


def load_npy_from_minio(
    minio_client: MinioStorageClient, user_id: str, object_name: str
) -> list[float]:
    """Download a .npy embedding file from MinIO and return as a Python list of floats."""
    data = minio_client.get_object_bytes(bucket=user_id, object_name=object_name)
    arr = np.load(io.BytesIO(data))
    return arr.flatten().tolist()


def make_qdrant_client(collection_name: str) -> QdrantStorageClient:
    """Create a Qdrant client for the given collection."""
    settings = get_settings()
    timeout = ADDITIONAL_KWARGS.get("qdrant_timeout", 30)
    config = QdrantConfig(
        host=settings.qdrant.host,
        port=settings.qdrant.port,
        collection_name=collection_name,
        timeout=timeout,
        use_grpc=True,
        prefer_grpc=True,
    )
    return QdrantStorageClient(config=config)


# Global SPLADE client (lazy-initialized)
_splade_client: SpladeClient | None = None


def _get_splade_client() -> SpladeClient:
    """Get or create the global SPLADE client."""
    global _splade_client
    if _splade_client is None:
        settings = get_settings()
        config = SpladeConfig(
            url=settings.triton.url,
            timeout=settings.triton.timeout,
        )
        _splade_client = SpladeClient(config)
    return _splade_client


def encode_sparse_vectors(texts: list[str]) -> list[SparseVector]:
    """Encode texts to sparse vectors using SPLADE via Triton.

    Uses a global client instance for efficiency.

    Args:
        texts: List of text strings to encode.

    Returns:
        List of SparseVector objects with indices and values.
    """
    if not texts:
        return []

    client = _get_splade_client()
    return client.encode(texts)


async def create_summary_artifact(
    task_name: str,
    collection_name: str,
    points_count: int,
    extra_info: dict | None = None,
) -> None:
    """Create a markdown and table artifact summarizing the indexing task."""
    if points_count == 0:
        return

    key = re.sub(r"[^a-z0-9-]", "-", task_name.lower())

    markdown_lines = [
        f"# {task_name} Summary\n",
        "| Field | Value |",
        "|-------|-------|",
        f"| **Collection** | `{collection_name}` |",
        f"| **Points Indexed** | `{points_count}` |",
    ]

    if extra_info:
        for k, v in extra_info.items():
            markdown_lines.append(f"| **{k}** | `{v}` |")

    markdown = "\n".join(markdown_lines)

    await acreate_markdown_artifact(
        key=key,
        markdown=markdown,
        description=f"{task_name} summary",
    )

    table_data = [{"Field": "Collection", "Value": collection_name}]
    table_data.append({"Field": "Points Indexed", "Value": str(points_count)})
    if extra_info:
        for k, v in extra_info.items():
            table_data.append({"Field": k, "Value": str(v)})

    await acreate_table_artifact(
        table=table_data,
        key=f"{key}-table",
        description=f"{task_name} stats",
    )