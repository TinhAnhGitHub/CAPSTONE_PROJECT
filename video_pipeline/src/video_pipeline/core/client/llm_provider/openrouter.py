import asyncio
import base64
from pathlib import Path

import aiohttp
from loguru import logger
from pydantic import BaseModel


class OpenRouterConfig(BaseModel):
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemini-2.5-flash-preview"
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: int = 120



class ChatMessage(BaseModel):
    role: str
    content: str


class OpenRouterResult(BaseModel):
    content: str
    model: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None


class OpenRouterClient:
    def __init__(self, config: OpenRouterConfig):
        self.base_url = config.base_url.rstrip("/")
        self.api_key = config.api_key
        self.model = config.model
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.timeout = config.timeout
        self._session: aiohttp.ClientSession | None = None


    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


    @staticmethod
    def encode_image(image_path: str | Path) -> str:
        path = Path(image_path)
        ext = path.suffix.lower().lstrip(".")
        mime = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
            "gif": "image/gif",
        }.get(ext, "image/png")
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def encode_image_bytes(data: bytes, mime: str = "image/jpeg") -> str:
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _make_image_part(image_url: str, detail: str = "auto") -> dict:
        return {
            "type": "image_url",
            "image_url": {"url": image_url, "detail": detail},
        }

    @staticmethod
    def _make_text_part(text: str) -> dict:
        """Build a text content part."""
        return {"type": "text", "text": text}

    async def check_health(self) -> bool:
        """Verify connectivity by listing models."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/models",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.error(f"OpenRouter health check failed: HTTP {resp.status}")
                    return False
                logger.success("OpenRouter API is healthy")
                return True
        except Exception as e:
            logger.error(f"OpenRouter health check error: {e}")
            return False

    async def _request(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **extra_params,
    ) -> OpenRouterResult | None:
        """
        Send a chat completion request to OpenRouter.

        Args:
            messages: OpenAI-format messages list
            model: override default model
            max_tokens: override default
            temperature: override default
            **extra_params: additional params (top_p, tools, response_format, etc.)
        """
        try:
            session = await self._get_session()
            payload: dict = {
                "model": model or self.model,
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature if temperature is not None else self.temperature,
            }
            payload.update(extra_params)

            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"OpenRouter HTTP {resp.status}: {body}")
                    return None
                data = await resp.json()

                choice = data["choices"][0]
                return OpenRouterResult(
                    content=choice["message"]["content"],
                    model=data.get("model"),
                    finish_reason=choice.get("finish_reason"),
                    usage=data.get("usage"),
                )
        except Exception:
            logger.exception("OpenRouter request failed")
            return None
        
    async def ainfer_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> OpenRouterResult | None:
        """
        Text-only inference.

        Args:
            prompt: user message
            system_prompt: optional system message
        """
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self._request(messages, **kwargs)

    async def ainfer_image(
        self,
        image_url: str,
        prompt: str = "Describe this image in detail.",
        system_prompt: str | None = None,
        detail: str = "auto",
        **kwargs,
    ) -> OpenRouterResult | None:
        """
        Image (+ optional text) inference.

        Args:
            image_url: data-URI or public URL
            prompt: text accompanying the image
            system_prompt: optional system message
            detail: "auto" | "low" | "high"
        """
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content_parts = [
            self._make_image_part(image_url, detail=detail),
            self._make_text_part(prompt),
        ]
        messages.append({"role": "user", "content": content_parts})
        return await self._request(messages, **kwargs)

    async def ainfer_multi_image(
        self,
        image_urls: list[str],
        prompt: str = "Describe these images.",
        system_prompt: str | None = None,
        detail: str = "auto",
        **kwargs,
    ) -> OpenRouterResult | None:
        """
        Multi-image inference (text + multiple images in one request).

        Args:
            image_urls: list of data-URIs or public URLs
            prompt: text accompanying the images
        """
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content_parts: list[dict] = [
            self._make_image_part(url, detail=detail) for url in image_urls
        ]
        content_parts.append(self._make_text_part(prompt))
        messages.append({"role": "user", "content": content_parts})
        return await self._request(messages, **kwargs)

    async def ainfer_chat(
        self,
        messages: list[dict],
        **kwargs,
    ) -> OpenRouterResult | None:
        """
        Raw chat completion — full control over messages.

        Useful for multi-turn conversations or custom content part layouts.
        Messages follow OpenAI format.
        """
        return await self._request(messages, **kwargs)


    async def batch_ainfer_text(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        max_concurrent: int = 10,
        **kwargs,
    ) -> list[OpenRouterResult | None]:
        """Run multiple text-only inferences concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _limited(prompt: str):
            async with semaphore:
                return await self.ainfer_text(prompt, system_prompt=system_prompt, **kwargs)

        return list(await asyncio.gather(*[_limited(p) for p in prompts]))

    async def batch_ainfer_image(
        self,
        image_urls: list[str],
        prompts: list[str] | str = "Describe this image in detail.",
        system_prompt: str | None = None,
        max_concurrent: int = 10,
        **kwargs,
    ) -> list[OpenRouterResult | None]:
        """
        Run multiple image inferences concurrently.

        Args:
            image_urls: list of image data-URIs or URLs
            prompts: a single prompt for all, or one prompt per image
        """
        if isinstance(prompts, str):
            prompts = [prompts] * len(image_urls)
        if len(prompts) != len(image_urls):
            raise ValueError("prompts and image_urls must have the same length")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _limited(url: str, prompt: str):
            async with semaphore:
                return await self.ainfer_image(
                    url, prompt=prompt, system_prompt=system_prompt, **kwargs
                )

        tasks = [_limited(u, p) for u, p in zip(image_urls, prompts)]
        return list(await asyncio.gather(*tasks))

    async def batch_ainfer_multimodal(
        self,
        requests: list[dict],
        max_concurrent: int = 10,
    ) -> list[OpenRouterResult | None]:
        """
        Run a heterogeneous batch of requests concurrently.

        Each request dict can contain:
            - "type": "text" | "image" | "multi_image" | "chat"
            - plus the corresponding kwargs for that method

        Example:
            [
                {"type": "text", "prompt": "Hello"},
                {"type": "image", "image_url": "data:...", "prompt": "What is this?"},
                {"type": "multi_image", "image_urls": [...], "prompt": "Compare these."},
                {"type": "chat", "messages": [...]},
            ]
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        dispatch = {
            "text": self.ainfer_text,
            "image": self.ainfer_image,
            "multi_image": self.ainfer_multi_image,
            "chat": self.ainfer_chat,
        }

        async def _limited(req: dict):
            async with semaphore:
                req_type = req.pop("type", "text")
                fn = dispatch.get(req_type)
                if fn is None:
                    logger.error(f"Unknown request type: {req_type}")
                    return None
                return await fn(**req)

        return list(
            await asyncio.gather(*[_limited(dict(r)) for r in requests])
        )