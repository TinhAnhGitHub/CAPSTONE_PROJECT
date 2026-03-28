from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
from typing import TypeVar

from loguru import logger
from pydantic import BaseModel

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock, MessageRole


T = TypeVar("T", bound=BaseModel)


class GeminiConfig(BaseModel):
    """Configuration for Gemini LLM client."""

    api_key: str | None = None
    model: str = "models/gemini-2.0-flash"
    max_tokens: int = 8192
    temperature: float = 0.7
    timeout: int = 120

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("GOOGLE_API_KEY", "")


class GeminiResult(BaseModel):
    """Result from Gemini inference."""

    content: str
    model: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None


class GeminiClient:
    """
    Pure LlamaIndex-based Google Gemini client with structured output support.

    This client wraps LlamaIndex's Google GenAI LLM and provides:
    - achat(): Raw chat completion with LlamaIndex ChatMessage format
    - as_structured_llm(): Structured output for Pydantic models
    - acomplete(): Text completion interface
    - stream_chat()/stream_complete(): Streaming interfaces
    - Multi-modal image support

    All methods use LlamaIndex's native APIs - no raw HTTP requests.
    """

    def __init__(self, config: GeminiConfig):
        self.config = config
        self.llm = GoogleGenAI(
            api_key=config.api_key or "",
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

    async def close(self):
        """Close any open resources (no-op for LlamaIndex clients)."""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def as_structured_llm(self, output_cls: type[T]):
        """
        Create a structured LLM for Pydantic output.

        Args:
            output_cls: Pydantic model class for structured output

        Returns:
            LLM instance configured for structured output
        """
        return self.llm.as_structured_llm(output_cls)

    async def achat(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> ChatMessage:
        """
        Send a chat completion using LlamaIndex format.

        Args:
            messages: List of LlamaIndex ChatMessage objects
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            ChatMessage response
        """
        response = await self.llm.achat(messages, **kwargs)
        return response.message

    def chat(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> ChatMessage:
        """
        Synchronous chat completion.

        Args:
            messages: List of LlamaIndex ChatMessage objects
            **kwargs: Additional parameters

        Returns:
            ChatMessage response
        """
        response = self.llm.chat(messages, **kwargs)
        return response.message

    async def acomplete(
        self,
        prompt: str,
        **kwargs,
    ) -> ChatMessage:
        """
        Text completion interface.

        Args:
            prompt: Text prompt for completion
            **kwargs: Additional parameters

        Returns:
            ChatMessage with completion
        """
        response = await self.llm.acomplete(prompt, **kwargs)
        return ChatMessage(role=MessageRole.ASSISTANT, content=response.text)

    def complete(
        self,
        prompt: str,
        **kwargs,
    ) -> ChatMessage:
        """
        Synchronous text completion.

        Args:
            prompt: Text prompt for completion
            **kwargs: Additional parameters

        Returns:
            ChatMessage with completion
        """
        response = self.llm.complete(prompt, **kwargs)
        return ChatMessage(role=MessageRole.ASSISTANT, content=response.text)

    def stream_chat(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ):
        """
        Streaming chat completion.

        Args:
            messages: List of LlamaIndex ChatMessage objects
            **kwargs: Additional parameters

        Yields:
            ChatMessage deltas
        """
        yield from self.llm.stream_chat(messages, **kwargs)

    def stream_complete(
        self,
        prompt: str,
        **kwargs,
    ):
        """
        Streaming text completion.

        Args:
            prompt: Text prompt for completion
            **kwargs: Additional parameters

        Yields:
            Text completion deltas
        """
        yield from self.llm.stream_complete(prompt, **kwargs)

    @staticmethod
    def encode_image(image_path: str | Path) -> str:
        """Encode image file to base64 data URI (utility method)."""
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
        """Encode image bytes to base64 data URI (utility method)."""
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    async def ainfer_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> GeminiResult | None:
        """
        Text-only inference using LlamaIndex achat.

        Args:
            prompt: User message
            system_prompt: Optional system message
            **kwargs: Additional parameters

        Returns:
            GeminiResult or None on error
        """
        try:
            messages: list[ChatMessage] = []
            if system_prompt:
                messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))
            messages.append(ChatMessage(role=MessageRole.USER, content=prompt))

            response = await self.achat(messages, **kwargs)
            return GeminiResult(
                content=response.content or "",
                model=self.config.model,
                finish_reason="stop",
            )
        except Exception as e:
            logger.exception(f"Text inference failed: {e}")
            return None

    async def ainfer_image(
        self,
        image_data: bytes | str,
        prompt: str = "Describe this image in detail.",
        system_prompt: str | None = None,
        is_data_uri: bool = False,
        **kwargs,
    ) -> GeminiResult | None:
        """
        Image (+ optional text) inference using LlamaIndex achat.

        Args:
            image_data: Image bytes or data URI string
            prompt: Text accompanying the image
            system_prompt: Optional system message
            is_data_uri: If True, image_data is a data URI string
            **kwargs: Additional parameters

        Returns:
            GeminiResult or None on error
        """
        try:
            messages: list[ChatMessage] = []
            if system_prompt:
                messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))

            if is_data_uri:
                mime, b64 = image_data.split(",", 1)  # type: ignore
                mime = mime.split(":")[1].split(";")[0]  # type: ignore
                image_bytes = base64.b64decode(b64)
            elif isinstance(image_data, bytes):
                image_bytes = image_data
                mime = "image/jpeg"
            else:
                
                image_bytes = Path(image_data).read_bytes()  # type: ignore
                mime = "image/jpeg" if str(image_data).endswith((".jpg", ".jpeg")) else "image/png"

            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    blocks=[
                        ImageBlock(image=image_bytes, image_mimetype=mime),  # type: ignore
                        TextBlock(text=prompt),
                    ],
                )
            )

            response = await self.achat(messages, **kwargs)
            return GeminiResult(
                content=response.content or "",
                model=self.config.model,
                finish_reason="stop",
            )
        except Exception as e:
            logger.exception(f"Image inference failed: {e}")
            return None

    async def ainfer_multi_image(
        self,
        image_data_list: list[bytes | str],
        prompt: str = "Describe these images.",
        system_prompt: str | None = None,
        **kwargs,
    ) -> GeminiResult | None:
        """
        Multi-image inference using LlamaIndex achat.

        Args:
            image_data_list: List of image bytes or data URIs
            prompt: Text accompanying the images
            system_prompt: Optional system message
            **kwargs: Additional parameters

        Returns:
            GeminiResult or None on error
        """
        try:
            messages: list[ChatMessage] = []
            if system_prompt:
                messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))

            blocks = []
            for image_data in image_data_list:
                if isinstance(image_data, str) and image_data.startswith("data:"):
                    # Parse data URI
                    mime, b64 = image_data.split(",", 1)
                    mime = mime.split(":")[1].split(";")[0]
                    image_bytes = base64.b64decode(b64)
                elif isinstance(image_data, bytes):
                    image_bytes = image_data
                    mime = "image/jpeg"
                else:
                    # Assume file path
                    image_bytes = Path(image_data).read_bytes()  # type: ignore
                    mime = (
                        "image/jpeg" if str(image_data).endswith((".jpg", ".jpeg")) else "image/png"
                    )

                blocks.append(ImageBlock(image=image_bytes, image_mimetype=mime))

            blocks.append(TextBlock(text=prompt))
            messages.append(ChatMessage(role=MessageRole.USER, blocks=blocks))

            response = await self.achat(messages, **kwargs)
            return GeminiResult(
                content=response.content or "",
                model=self.config.model,
                finish_reason="stop",
            )
        except Exception as e:
            logger.exception(f"Multi-image inference failed: {e}")
            return None

    async def ainfer_chat(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> GeminiResult | None:
        """
        Chat completion using LlamaIndex ChatMessage format.

        Args:
            messages: List of LlamaIndex ChatMessage objects
            **kwargs: Additional parameters

        Returns:
            GeminiResult or None on error
        """
        try:
            response = await self.achat(messages, **kwargs)
            return GeminiResult(
                content=response.content or "",
                model=self.config.model,
                finish_reason="stop",
            )
        except Exception as e:
            logger.exception(f"Chat completion failed: {e}")
            return None

    async def batch_ainfer_text(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        max_concurrent: int = 10,
        **kwargs,
    ) -> list[GeminiResult | None]:
        """Run multiple text-only inferences concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _limited(prompt: str):
            async with semaphore:
                return await self.ainfer_text(prompt, system_prompt=system_prompt, **kwargs)

        return list(await asyncio.gather(*[_limited(p) for p in prompts]))

    async def batch_ainfer_image(
        self,
        image_data_list: list[bytes | str],
        prompts: list[str] | str = "Describe this image in detail.",
        system_prompt: str | None = None,
        max_concurrent: int = 10,
        **kwargs,
    ) -> list[GeminiResult | None]:
        """
        Run multiple image inferences concurrently.

        Args:
            image_data_list: List of image bytes or data URIs
            prompts: Single prompt for all images, or list of prompts
            system_prompt: Optional system message
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional parameters

        Returns:
            List of GeminiResult or None for each image
        """
        if isinstance(prompts, str):
            prompts = [prompts] * len(image_data_list)
        if len(prompts) != len(image_data_list):
            raise ValueError("prompts and image_data_list must have the same length")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _limited(image_data: bytes | str, prompt: str) -> GeminiResult | None:
            async with semaphore:
                return await self.ainfer_image(
                    image_data, prompt=prompt, system_prompt=system_prompt, **kwargs
                )

        tasks = [_limited(img, p) for img, p in zip(image_data_list, prompts)]
        return list(await asyncio.gather(*tasks))

    async def batch_ainfer_chat(
        self,
        messages_list: list[list[ChatMessage]],
        max_concurrent: int = 10,
        **kwargs,
    ) -> list[GeminiResult | None]:
        """
        Run multiple chat completions concurrently.

        Args:
            messages_list: List of message lists (each list is a conversation)
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional parameters

        Returns:
            List of GeminiResult or None for each conversation
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _limited(messages: list[ChatMessage]):
            async with semaphore:
                return await self.ainfer_chat(messages, **kwargs)

        return list(await asyncio.gather(*[_limited(msgs) for msgs in messages_list]))