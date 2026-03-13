from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
from typing import TypeVar

from loguru import logger
from pydantic import BaseModel

from llama_index.llms.openrouter import OpenRouter as LlamaIndexOpenRouter
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock, MessageRole
from llama_index.core.multi_modal_llms import MultiModalLLM


T = TypeVar("T", bound=BaseModel)


class OpenRouterConfig(BaseModel):
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemini-2.5-flash-preview"
    max_tokens: int = 4096
    temperature: float = 0.6
    timeout: int = 120

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("OPENROUTER_API_KEY", "")


class OpenRouterResult(BaseModel):
    content: str
    model: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None


class OpenRouterClient:
    """
    Pure LlamaIndex-based OpenRouter client with structured output support.

    This client wraps LlamaIndex's OpenRouter LLM and provides:
    - achat(): Raw chat completion with LlamaIndex ChatMessage format
    - as_structured_llm(): Structured output for Pydantic models
    - acomplete(): Text completion interface
    - stream_chat()/stream_complete(): Streaming interfaces

    All methods use LlamaIndex's native APIs - no raw HTTP requests.
    """

    def __init__(self, config: OpenRouterConfig):
        self.config = config
        self.llm = LlamaIndexOpenRouter(
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

    def _extract_usage(self, response) -> dict:
        """Extract usage info from LlamaIndex ChatResponse."""
        usage = {}

        # From additional_kwargs (always available)
        if hasattr(response, "additional_kwargs"):
            usage["prompt_tokens"] = response.additional_kwargs.get("prompt_tokens", 0)
            usage["completion_tokens"] = response.additional_kwargs.get("completion_tokens", 0)
            usage["total_tokens"] = response.additional_kwargs.get("total_tokens", 0)

        # From raw.usage (more detailed, includes cost)
        if hasattr(response, "raw") and response.raw:
            raw = response.raw
            if hasattr(raw, "usage") and raw.usage:
                raw_usage = raw.usage
                usage["prompt_tokens"] = getattr(raw_usage, "prompt_tokens", usage.get("prompt_tokens", 0))
                usage["completion_tokens"] = getattr(raw_usage, "completion_tokens", usage.get("completion_tokens", 0))
                usage["total_tokens"] = getattr(raw_usage, "total_tokens", usage.get("total_tokens", 0))
                usage["cost"] = getattr(raw_usage, "cost", 0.0)

        return usage

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
    ) -> OpenRouterResult | None:
        """
        Text-only inference using LlamaIndex acomplete.

        Args:
            prompt: User message
            system_prompt: Optional system message
            **kwargs: Additional parameters

        Returns:
            OpenRouterResult or None on error
        """
        try:
            messages: list[ChatMessage] = []
            if system_prompt:
                messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))
            messages.append(ChatMessage(role=MessageRole.USER, content=prompt))

            full_response = await self.llm.achat(messages, **kwargs)
            usage = self._extract_usage(full_response)

            return OpenRouterResult(
                content=full_response.message.content or "",
                model=self.config.model,
                finish_reason="stop",
                usage=usage,
            )
        except Exception as e:
            logger.exception(f"Text inference failed: {e}")
            return None

    async def ainfer_image(
        self,
        image_data: bytes | str,
        prompt: str = "Describe this image in detail.",
        system_prompt: str | None = None,
        **kwargs,
    ) -> OpenRouterResult | None:
        """
        Image (+ optional text) inference using LlamaIndex achat.

        Args:
            image_data: Image bytes or data URI string
            prompt: Text accompanying the image
            system_prompt: Optional system message
            **kwargs: Additional parameters

        Returns:
            OpenRouterResult or None on error
        """
        try:
            messages: list[ChatMessage] = []
            if system_prompt:
                messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))

            # Prepare image block
            if isinstance(image_data, str) and image_data.startswith("data:"):
                # Auto-detect data URI
                mime, b64 = image_data.split(",", 1) #type:ignore
                mime = mime.split(":")[1].split(";")[0] #type:ignore
                image_bytes = base64.b64decode(b64)
            elif isinstance(image_data, bytes):
                image_bytes = image_data
                mime = "image/jpeg"
            else:
                # Assume it's a file path
                image_bytes = Path(image_data).read_bytes() #type:ignore
                mime = "image/jpeg" if str(image_data).endswith((".jpg", ".jpeg")) else "image/png"

            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    blocks=[
                        ImageBlock(image=image_bytes, image_mimetype=mime), #type:ignore
                        TextBlock(text=prompt),
                    ],
                )
            )

            full_response = await self.llm.achat(messages, **kwargs)
            usage = self._extract_usage(full_response)

            return OpenRouterResult(
                content=full_response.message.content or "",
                model=self.config.model,
                finish_reason="stop",
                usage=usage,
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
    ) -> OpenRouterResult | None:
        """
        Multi-image inference using LlamaIndex achat.

        Args:
            image_data_list: List of image bytes or data URIs
            prompt: Text accompanying the images
            system_prompt: Optional system message
            **kwargs: Additional parameters

        Returns:
            OpenRouterResult or None on error
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
                    image_bytes = Path(image_data).read_bytes() #type:ignore
                    mime = (
                        "image/jpeg" if str(image_data).endswith((".jpg", ".jpeg")) else "image/png"
                    )

                blocks.append(ImageBlock(image=image_bytes, image_mimetype=mime))

            blocks.append(TextBlock(text=prompt))
            messages.append(ChatMessage(role=MessageRole.USER, blocks=blocks))

            full_response = await self.llm.achat(messages, **kwargs)
            usage = self._extract_usage(full_response)

            return OpenRouterResult(
                content=full_response.message.content or "",
                model=self.config.model,
                finish_reason="stop",
                usage=usage,
            )
        except Exception as e:
            logger.exception(f"Multi-image inference failed: {e}")
            return None

    async def ainfer_chat(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> OpenRouterResult | None:
        """
        Chat completion using LlamaIndex ChatMessage format.

        Args:
            messages: List of LlamaIndex ChatMessage objects
            **kwargs: Additional parameters

        Returns:
            OpenRouterResult or None on error
        """
        try:
            full_response = await self.llm.achat(messages, **kwargs)
            usage = self._extract_usage(full_response)

            return OpenRouterResult(
                content=full_response.message.content or "",
                model=self.config.model,
                finish_reason="stop",
                usage=usage,
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
    ) -> list[OpenRouterResult | None]:
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
    ) -> list[OpenRouterResult | None]:
        """
        Run multiple image inferences concurrently.

        Args:
            image_data_list: List of image bytes or data URIs
            prompts: Single prompt for all images, or list of prompts
            system_prompt: Optional system message
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional parameters

        Returns:
            List of OpenRouterResult or None for each image
        """
        if isinstance(prompts, str):
            prompts = [prompts] * len(image_data_list)
        if len(prompts) != len(image_data_list):
            raise ValueError("prompts and image_data_list must have the same length")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _limited(image_data: bytes | str, prompt: str) -> OpenRouterResult | None:
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
    ) -> list[OpenRouterResult | None]:
        """
        Run multiple chat completions concurrently.

        Args:
            messages_list: List of message lists (each list is a conversation)
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional parameters

        Returns:
            List of OpenRouterResult or None for each conversation
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _limited(messages: list[ChatMessage]):
            async with semaphore:
                return await self.ainfer_chat(messages, **kwargs)

        return list(await asyncio.gather(*[_limited(msgs) for msgs in messages_list]))
