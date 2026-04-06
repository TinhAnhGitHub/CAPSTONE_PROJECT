"""Inference clients for embedding services (QwenVL, MMBert)."""

from __future__ import annotations

import asyncio
import base64
from typing import Optional

import aiohttp
import httpx
import numpy as np
from loguru import logger

from videodeepsearch.clients.inference.schema import MMBertConfig, QwenVLEmbeddingConfig, SpladeConfig


class QwenVLEmbeddingClient:
    """Client for QwenVL multimodal embedding service.

    Supports text, image, and video embeddings.
    """

    def __init__(self, config: QwenVLEmbeddingConfig):
        """Initialize the QwenVL client.

        Args:
            config: Configuration with base_url
        """
        self.base_url = config.base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()

    async def _get_embedding(
        self,
        prompt_string: str,
        multimodal_data: list[str] | None = None,
    ) -> list[float]:
        """Send request to the server and return a normalized embedding.

        Args:
            prompt_string: The prompt to send
            multimodal_data: Optional list of base64 encoded images

        Returns:
            Normalized embedding vector
        """
        if multimodal_data is None:
            multimodal_data = []

        payload = {
            "content": [
                {
                    "prompt_string": prompt_string,
                    "multimodal_data": multimodal_data,
                }
            ]
        }

        client = await self._get_client()
        response = await client.post(self.base_url, json=payload)
        response.raise_for_status()

        tokens = np.array(response.json()[0]["embedding"])
        vec = tokens[-1]
        vec = vec / np.linalg.norm(vec)

        return vec.tolist()

    async def ainfer_text(self, texts: list[str]) -> list[list[float]]:
        """Process a list of texts concurrently and return their embeddings.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        tasks = []
        for text in texts:
            prompt = f"Represent for retrieval: {text}"
            tasks.append(self._get_embedding(prompt))
        return await asyncio.gather(*tasks)

    async def _infer_single_image(
        self, image_bytes: bytes, text: str = ""
    ) -> list[float]:
        """Process a single image as a standalone request.

        Args:
            image_bytes: Raw image bytes
            text: Optional text to include in the prompt

        Returns:
            Embedding vector
        """
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        prompt_string = "Image:\n<__media__>\nRepresent for retrieval."
        if text:
            prompt_string = f"Image:\n<__media__>\nRepresent for retrieval: {text}"

        return await self._get_embedding(prompt_string, [b64_image])

    async def ainfer_image(
        self, images: list[bytes], text: str = ""
    ) -> list[list[float]]:
        """Process multiple images independently.

        Args:
            images: List of image bytes
            text: Optional text to include in the prompt

        Returns:
            List of embedding vectors (one per image)
        """
        tasks = [self._infer_single_image(img, text) for img in images]
        return await asyncio.gather(*tasks)

    async def ainfer_video(
        self, frames: list[bytes], text: str = ""
    ) -> list[float]:
        """Process multiple video frames into a SINGLE embedding.

        Args:
            frames: List of frame bytes
            text: Optional text to include in the prompt

        Returns:
            Single embedding vector for the video
        """
        b64_frames = [base64.b64encode(frame).decode("utf-8") for frame in frames]
        media_tags = " ".join(["<__media__>"] * len(b64_frames))

        prompt_string = f"Video:\n{media_tags}\nRepresent for later on retrieval."
        if text:
            prompt_string += f" {text}"

        return await self._get_embedding(prompt_string, b64_frames)


class MMBertClient:
    """Client for MMBert text embedding service.

    Specialized for Vietnamese text embeddings via llama.cpp server.
    """

    def __init__(self, config: MMBertConfig):
        """Initialize the MMBert client.

        Args:
            config: Configuration with base_url and model_name
        """
        self.base_url = config.base_url.rstrip("/")
        self.model_name = config.model_name
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def check_health(self) -> bool:
        """Check if the llama.cpp server is up and reports a model.

        Returns:
            True if the service is healthy, False otherwise
        """
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/v1/models",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    logger.error(f"MMBert health check failed: HTTP {resp.status}")
                    return False
                data = await resp.json()
                models = [m["id"] for m in data.get("data", [])]
                logger.success(f"MMBert server ready. Models: {models}")
                return True
        except Exception as e:
            logger.error(f"MMBert health check error: {e}")
            return False

    async def ainfer(self, texts: list[str]) -> list[list[float]] | None:
        """Embed a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors, or None on failure
        """
        try:
            session = await self._get_session()
            payload = {"model": self.model_name, "input": texts}
            async with session.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"MMBert inference HTTP {resp.status}: {body}")
                    return None
                data = await resp.json()
                embeddings = [item["embedding"] for item in data["data"]]
                return embeddings
        except Exception:
            logger.exception("MMBert ainfer failed")
            return None

    async def batch_ainfer(
        self, text_chunks: list[list[str]]
    ) -> list[list[list[float]] | None]:
        """Embed multiple batches of texts concurrently.

        Args:
            text_chunks: List of text lists (batches)

        Returns:
            List of embedding results (one per batch)
        """
        tasks = [self.ainfer(chunk) for chunk in text_chunks]
        return list(await asyncio.gather(*tasks))


class SpladeClient:
    """Client for SPLADE sparse embeddings via Triton Inference Server.

    SPLADE produces sparse vectors suitable for hybrid search with Qdrant.
    Uses gRPC protocol to communicate with Triton.
    """

    def __init__(self, config: SpladeConfig):
        """Initialize the SPLADE client.

        Args:
            config: Configuration with url, model_name, timeout, etc.
        """
        from tritonclient.grpc import InferenceServerClient

        self.config = config
        self._client: InferenceServerClient | None = None

    def _get_client(self):
        """Get or create the Triton gRPC client."""
        if self._client is None:
            from tritonclient.grpc import InferenceServerClient

            self._client = InferenceServerClient(
                url=self.config.url,
                verbose=self.config.verbose,
            )
        return self._client

    def close(self) -> None:
        """Close the Triton client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "SpladeClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def check_health(self) -> bool:
        """Check if Triton server is live and model is ready.

        Returns:
            True if healthy, False otherwise
        """
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

    def _encode_batch(self, texts: list[str]) -> list[dict[int, float]]:
        """Encode a single batch of texts to sparse vectors.

        Args:
            texts: List of texts (must be <= max_batch_size)

        Returns:
            List of sparse vectors as {token_index: value}
        """
        import pickle
        from tritonclient.grpc import InferInput, InferRequestedOutput

        if not texts:
            return []

        client = self._get_client()

        text_input = InferInput("TEXT", [len(texts), 1], "BYTES")
        text_input.set_data_from_numpy(np.array([[t] for t in texts], dtype=object))

        outputs = [
            InferRequestedOutput("INDICES"),
            InferRequestedOutput("VALUES"),
        ]

        result = client.infer(
            model_name=self.config.model_name,
            inputs=[text_input],
            outputs=outputs,
            timeout=self.config.timeout,
        )

        indices_batch = result.as_numpy("INDICES")
        values_batch = result.as_numpy("VALUES")

        sparse_vectors: list[dict[int, float]] = []
        for i in range(len(texts)):
            indices = pickle.loads(indices_batch[i]).tolist()
            values = pickle.loads(values_batch[i]).tolist()
            # Convert to {index: value} format for Qdrant
            sparse_vectors.append({int(idx): float(val) for idx, val in zip(indices, values)})

        return sparse_vectors

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        """Encode texts to sparse vectors, handling batching.

        Args:
            texts: List of text strings

        Returns:
            List of sparse vectors as {token_index: value}
        """
        if not texts:
            return []

        max_batch = self.config.max_batch_size

        if len(texts) <= max_batch:
            return self._encode_batch(texts)

        all_vectors: list[dict[int, float]] = []
        for i in range(0, len(texts), max_batch):
            batch = texts[i : i + max_batch]
            batch_vectors = self._encode_batch(batch)
            all_vectors.extend(batch_vectors)

        logger.debug(
            f"[SpladeClient] Encoded {len(texts)} text(s) in {(len(texts) + max_batch - 1) // max_batch} batch(es)"
        )
        return all_vectors

    async def aencode(self, texts: list[str]) -> list[dict[int, float]]:
        """Async wrapper for encode.

        Args:
            texts: List of text strings

        Returns:
            List of sparse vectors as {token_index: value}
        """
        return self.encode(texts)


__all__ = [
    "QwenVLEmbeddingClient",
    "MMBertClient",
    "SpladeClient",
]