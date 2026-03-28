import asyncio
import base64
from pathlib import Path

import aiohttp
from loguru import logger
from pydantic import BaseModel



def normalize_ocr(text: str) -> str:
    t = text.strip()

    markers = [
        "![image](image_1.png)",
        "![image](image.png)",
        "![image]",
        "\n![image](image_1.png)",
    ]

    for mark in markers:
        t = t.replace(mark, '')

    return t


class LightONOCRConfig(BaseModel):
    model_name: str = "ocr_lighton"
    base_url: str = "http://localhost:8011"
    system_prompt: str = (
        "Extract all text from the image"
        "appears. Return only the extracted text with no commentary. If no text, return empty"
    )
    max_tokens: int = 2048


class LightONOCRClient:
    def __init__(self, config: LightONOCRConfig):
        self.base_url = config.base_url.rstrip("/")
        self.model_name = config.model_name
        self.system_prompt = config.system_prompt
        self.max_tokens = config.max_tokens
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    @staticmethod
    def _encode_image(image_path: str | Path) -> tuple[str, str]:
        """Return (base64_data, mime_type) for an image file."""
        path = Path(image_path)
        ext = path.suffix.lower().lstrip(".")
        mime = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
        }.get(ext, "image/png")
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return b64, mime

    async def check_health(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/v1/models", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    logger.error(f"LightON OCR health check failed: HTTP {resp.status}")
                    return False
                data = await resp.json()
                models = [m["id"] for m in data.get("data", [])]
                logger.success(f"LightON OCR server ready. Models: {models}")
                return True
        except Exception as e:
            logger.error(f"LightON OCR health check error: {e}")
            return False

    async def _ocr_one(self, image_path: str | Path) -> str:
        try:
            b64, mime = self._encode_image(image_path)
            session = await self._get_session()
            payload = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            },
                        ],
                    },
                ],
            }
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"LightON OCR HTTP {resp.status}: {body}")
                    raise RuntimeError(
                        f"OCR HTTP {resp.status}: {body[:500]}"
                    )
                data = await resp.json()
                return normalize_ocr(data["choices"][0]["message"]["content"])
        except Exception:
            logger.exception(f"LightON OCR failed for {image_path}")
            raise

    async def ainfer(self, image_paths: list[str | Path]) -> list[str]:
        try:
            tasks = [self._ocr_one(p) for p in image_paths]
            return list(await asyncio.gather(*tasks))
        except Exception:
            logger.exception("LightON OCR ainfer failed")
            raise

    async def batch_ainfer(
        self, path_groups: list[list[str | Path]]
    ) -> list[list[str]]:
        tasks = [self.ainfer(group) for group in path_groups]
        return list(await asyncio.gather(*tasks))
