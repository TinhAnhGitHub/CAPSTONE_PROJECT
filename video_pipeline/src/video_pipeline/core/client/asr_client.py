from pathlib import Path
import asyncio
import aiohttp
from loguru import logger
from pydantic import BaseModel
from typing import Any


class QwenASRConfig(BaseModel):
    model_name: str = "Qwen/Qwen3-ASR-0.6B"
    timeout: int = 300
    max_tokens: int = 256
    temperature: float = 0.01


class QwenASRClient:
    def __init__(self, base_url: str, asr_config: QwenASRConfig):
        """
        Initialize Qwen ASR client

        Args:
            base_url: Base URL of vLLM server (e.g., "http://localhost:8020")
            asr_config: Configuration for ASR client
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = asr_config.model_name
        self.timeout = asr_config.timeout
        self.max_tokens = asr_config.max_tokens
        self.temperature = asr_config.temperature

        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def check_server_health(self) -> bool:
        try:
            session = await self._get_session()

            async with session.get(f"{self.base_url}/health") as response:
                if response.status != 200:
                    logger.error(
                        f"Server health check failed with status {response.status}"
                    )
                    return False

            async with session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m.get("id") for m in data.get("data", [])]

                    if self.model_name not in models:
                        logger.error(
                            f"Model {self.model_name} not found. Available: {models}"
                        )
                        return False

                    logger.success(f"Server and model {self.model_name} are ready")
                    return True
                else:
                    logger.warning("Could not check model availability")
                    return True

        except Exception as e:
            logger.error(f"Error checking server health: {e}")
            return False

    async def transcribe(
        self, audio_path: Path, language: str | None = None
    ) -> dict[str, str] | None:
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_path=}")
                return None

            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            session = await self._get_session()
            data = aiohttp.FormData()
            data.add_field(
                "file", audio_bytes, filename=audio_path.name, content_type="audio/wav"
            )
            data.add_field("model", self.model_name)
            if language:
                data.add_field("language", language)

            async with session.post(
                f"{self.base_url}/v1/audio/transcriptions", data=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.success(f"Autio path: {audio_path=} transcribe complete")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Transcription failed: {response.status} - {error_text}"
                    )
                    return None

        except Exception:
            logger.exception(f"Transcription failed for {audio_path}")
            return None

    async def batch_transcribe(
        self, audio_paths: list[Path], languages: list[str | None] | None = None
    ) -> list[dict[str, str] | None]:
        if languages is None:
            languages = [None] * len(audio_paths)  # type:ignore

        tasks = [
            self.transcribe(audio_path, lang)
            for audio_path, lang in zip(audio_paths, languages)  # type:ignore
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed_results = []
        for audio_path, result in zip(audio_paths, results):
            if isinstance(result, Exception):
                logger.error(f"Batch transcription failed for {audio_path=}: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)

        return processed_results

    async def get_model_info(self) -> dict[str, Any] | None:
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/v1/models",
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get model info: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return None
