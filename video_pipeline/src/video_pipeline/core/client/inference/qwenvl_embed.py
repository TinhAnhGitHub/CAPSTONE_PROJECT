import asyncio
import base64
import httpx
from typing import List
from pydantic import BaseModel
from loguru import logger


class QwenVLEmbeddingConfig(BaseModel):
    model_name: str = "qwen3-vl-embedding-2b"
    base_url: str = "http://localhost:8010"


class QwenVLEmbeddingClient:
    def __init__(self, config: QwenVLEmbeddingConfig):
        self.base_url = config.base_url.rstrip("/")
        self.model_name = config.model_name
        self._client: httpx.AsyncClient | None = None

    async def check_health(self) -> bool:
        try:
            client = await self._get_client()
            # Checking the models endpoint is the standard way to verify readiness
            resp = await client.get(f"{self.base_url}/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                models = [m["id"] for m in data.get("data", [])]
                if self.model_name in models:
                    logger.success(
                        f"QwenVL healthy. Model '{self.model_name}' is loaded."
                    )
                    return True
                logger.warning(
                    f"Server up, but model '{self.model_name}' not found in {models}"
                )
                return False
            logger.error(f"Health check failed with status: {resp.status_code}")
            return False
        except Exception as e:
            logger.error(f"Could not connect to QwenVL server: {e}")
            return False

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()

    async def ainfer_text(self, texts: List[str]) -> List[List[float]]:
        client = await self._get_client()
        payload = {"model": self.model_name, "input": texts, "encoding_format": "float"}

        response = await client.post(f"{self.base_url}/v1/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    async def _infer_single_image(
        self, image_bytes: bytes, text: str = ""
    ) -> List[float]:
        """
        Internal helper to process a single image as a standalone request.
        """
        client = await self._get_client()
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpg;base64,{b64_image}"},
            }
        ]
        if text:
            content.append({"type": "text", "text": text})

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
        }

        response = await client.post(f"{self.base_url}/pooling", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["data"][0]

    async def ainfer_image(
        self, images: List[bytes], text: str = ""
    ) -> List[List[float]]:
        tasks = [self._infer_single_image(img, text) for img in images]
        return await asyncio.gather(*tasks)
