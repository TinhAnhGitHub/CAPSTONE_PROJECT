import asyncio
import base64
import httpx
import numpy as np
from typing import List, Optional
from pydantic import BaseModel
from loguru import logger

class QwenVLEmbeddingConfig(BaseModel):
    base_url: str
    timeout: float = 300.0  # 5 minutes default for video embeddings
    max_retries: int = 3

class QwenVLEmbeddingClient:
    def __init__(self, config: QwenVLEmbeddingConfig):
        self.base_url = config.base_url.rstrip("/")
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            # Use transport with retries for resilience
            transport = httpx.AsyncHTTPTransport(retries=self.max_retries)
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                transport=transport,
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()

    async def _get_embedding(self, prompt_string: str, multimodal_data: List[str] | None = None) -> List[float]:
        """Core async method to send request to the server and return a normalized embedding."""
        if multimodal_data is None:
            multimodal_data = []

        payload = {
            "content": [
                {
                    "prompt_string": prompt_string,
                    "multimodal_data": multimodal_data
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

    async def ainfer_text(self, texts: List[str]) -> List[List[float]]:
        """Process a list of texts concurrently and return their embeddings."""
        tasks = []
        for text in texts:
            prompt = f"Represent for retrieval: {text}"
            tasks.append(self._get_embedding(prompt))
        return await asyncio.gather(*tasks)

    async def _infer_single_image(self, image_bytes: bytes, text: str = "") -> List[float]:
        """Internal helper to process a single image as a standalone request."""
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        prompt_string = "Image:\n<__media__>\nRepresent for retrieval."
        if text:
            prompt_string = f"Image:\n<__media__>\nRepresent for retrieval: {text}"

        return await self._get_embedding(prompt_string, [b64_image])

    async def ainfer_image(self, images: List[bytes], text: str = "") -> List[List[float]]:
        """Process multiple images independently, returning one embedding per image."""
        tasks = [self._infer_single_image(img, text) for img in images]
        return await asyncio.gather(*tasks)

    async def ainfer_video(self, frames: List[bytes], text: str = "") -> List[float]:
        """
        Process multiple video frames into a SINGLE embedding,
        mirroring the logic in your MAIN PIPELINE.
        """
        b64_frames = [base64.b64encode(frame).decode('utf-8') for frame in frames]
        media_tags = " ".join(["<__media__>"] * len(b64_frames))

        prompt_string = f"Video:\n{media_tags}\nRepresent for later on retrieval."
        if text:
            prompt_string += f" {text}"

        return await self._get_embedding(prompt_string, b64_frames)