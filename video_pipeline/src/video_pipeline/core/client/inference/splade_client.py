"""SPLADE sparse embedding client using Triton Inference Server."""

from __future__ import annotations

from typing import Any, cast
import pickle
import numpy as np
from loguru import logger
from pydantic import BaseModel
from qdrant_client.models import SparseVector
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput, InferResult 


class SpladeConfig(BaseModel):
    """Configuration for SPLADE Triton client."""

    url: str = "localhost:8001"
    model_name: str = "splade"
    timeout: int = 30
    verbose: bool = False
    max_batch_size: int = 32  # Triton SPLADE model batch size limit


class SpladeClient:
    def __init__(self, config: SpladeConfig):
        self.config = config
        self._client: InferenceServerClient | None = None

    def _get_client(self) -> InferenceServerClient:
        if self._client is None:
            self._client = InferenceServerClient(
                url=self.config.url,
                verbose=self.config.verbose,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "SpladeClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def check_health(self) -> bool:
        try:
            client = self._get_client()

            if not client.is_server_live():
                logger.error("[SpladeClient] Triton server is not live")
                return False

            if not client.is_model_ready(self.config.model_name):
                logger.error(
                    f"[SpladeClient] Model '{self.config.model_name}' is not ready"
                )
                return False

            logger.success(
                f"[SpladeClient] Triton server healthy, model '{self.config.model_name}' ready"
            )
            return True

        except Exception as e:
            logger.error(f"[SpladeClient] Health check failed: {e}")
            return False

    def _encode_batch(self, texts: list[str]) -> list[SparseVector]:
        """Encode a single batch of texts (must be <= max_batch_size)."""
        if not texts:
            return []

        client = self._get_client()

        text_input = InferInput("TEXT", [len(texts), 1], "BYTES")
        text_input.set_data_from_numpy(np.array([[t] for t in texts], dtype=object))

        outputs = [
            InferRequestedOutput("INDICES"),
            InferRequestedOutput("VALUES"),
        ]

        try:
            result = client.infer(
                model_name=self.config.model_name,
                inputs=[text_input],
                outputs=outputs,
                timeout=self.config.timeout,
            )
        except Exception as e:
            logger.error(f"[SpladeClient] Inference failed: {e}")
            raise RuntimeError(f"SPLADE inference failed: {e}") from e

        if result is None:
            logger.error(f"[SpladeClient] Inference failed: result return is None for some reason")
            raise RuntimeError(f"SPLADE inference failed. Result return None")

        indices_batch = result.as_numpy("INDICES")
        values_batch = result.as_numpy("VALUES")

        assert indices_batch is not None, "[SpladeClient] Indices batch is None for some reason"
        assert values_batch is not None, "[SpladeClient] Value batch is None for some reason"

        sparse_vectors = []
        for i in range(len(texts)):
            indices = pickle.loads(indices_batch[i]).tolist()
            values = pickle.loads(values_batch[i]).tolist()

            sparse_vectors.append(
                SparseVector(
                    indices=indices,
                    values=values,
                )
            )

        return sparse_vectors

    def encode(self, texts: list[str]) -> list[SparseVector]:
        """Encode texts to sparse vectors, handling batching for large inputs."""
        if not texts:
            return []

        max_batch = self.config.max_batch_size

        if len(texts) <= max_batch:
            return self._encode_batch(texts)

        all_vectors: list[SparseVector] = []
        for i in range(0, len(texts), max_batch):
            batch = texts[i : i + max_batch]
            batch_vectors = self._encode_batch(batch)
            all_vectors.extend(batch_vectors)

        logger.debug(
            f"[SpladeClient] Encoded {len(texts)} text(s) in {(len(texts) + max_batch - 1) // max_batch} batch(es)"
        )
        return all_vectors

    async def aencode(self, texts: list[str]) -> list[SparseVector]:
        return self.encode(texts)

