import json
import asyncio
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel


class QwenASRConfig(BaseModel):
    model_name: str = "Qwen/Qwen3-ASR-0.6B"
    api_key: str = "EMPTY" 


class QwenASRClient:
    def __init__(self, client_url: str, config: QwenASRConfig):
        self.client_url = client_url
        self.model_name = config.model_name
        
        self.client = AsyncOpenAI(
            base_url=self.client_url, 
            api_key=config.api_key
        )

    async def check_server_health(self) -> bool:
        try:
            models = await self.client.models.list()
            available_models = [m.id for m in models.data]
            
            if self.model_name not in available_models:
                logger.error(f"ASR model '{self.model_name}' not found. Available: {available_models}")
                return False
                
            logger.success("Qwen ASR server is ready")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def _transcribe_single(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            response = await self.client.audio.transcriptions.create(
                model=self.model_name,
                file=f,
                response_format="text"
            )
        return json.loads(response)

    async def ainfer(self, audio_paths: list[str]) -> list[dict] | None:
        """
        Transcribe a list of audio files concurrently.
        """
        try:
            tasks = [self._transcribe_single(p) for p in audio_paths]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            final_output = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to transcribe {audio_paths[i]}: {result}")
                    final_output.append("") 
                else:
                    final_output.append(result)
                    
            return final_output
            
        except Exception:
            logger.exception("ASR async inference failed")
            return None

    async def close(self):
        await self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()