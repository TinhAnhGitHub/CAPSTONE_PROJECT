import asyncio

import aiohttp
from loguru import logger
from pydantic import BaseModel


class MMBertConfig(BaseModel):
    model_name: str = "mmbert"  # served-model-name in llama.cpp
    base_url: str = "http://localhost:8100"


class MMBertClient:
    def __init__(self, config: MMBertConfig):
        self.base_url = config.base_url.rstrip("/")
        self.model_name = config.model_name
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def check_health(self) -> bool:
        """Return True if the llama.cpp server is up and reports a model."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/v1/models", timeout=aiohttp.ClientTimeout(total=5)
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
        tasks = [self.ainfer(chunk) for chunk in text_chunks]
        return list(await asyncio.gather(*tasks))
