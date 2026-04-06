from __future__ import annotations

import asyncio
import base64

from llm_json import json
import os
import re
from langchain_core.output_parsers import PydanticOutputParser
from pathlib import Path
from typing import TypeVar
from pydantic import SecretStr
from loguru import logger
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain_core.output_parsers import PydanticOutputParser


T = TypeVar("T", bound=BaseModel)


class OpenRouterConfig(BaseModel):
    api_key: SecretStr | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemini-2.5-flash-preview"
    max_tokens: int = 4096
    temperature: float = 0.6
    timeout: int = 120

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = SecretStr(os.getenv("OPENROUTER_API_KEY", ""))


class OpenRouterResult(BaseModel):
    content: str
    model: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None


class OpenRouterClient:
    """
    LangChain-based OpenRouter client with structured output support.

    This client wraps LangChain's ChatOpenAI (pointed at OpenRouter) and provides:
    - achat(): Raw chat completion with LangChain BaseMessage format
    - as_structured_llm(): Structured output for Pydantic models
    - acomplete(): Text completion interface
    - stream_chat()/stream_complete(): Streaming interfaces

    All methods use LangChain's native APIs.
    """

    def __init__(self, config: OpenRouterConfig):
        self.config = config
        self.llm = ChatOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.model,
            max_completion_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
            model_kwargs={
                "extra_body": {
                    "reasoning": {"effort": "none"}
                }
            },
        )

    async def close(self):
        """Close any open resources (no-op for LangChain clients)."""
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def as_structured_llm(self, output_cls: type[T], max_retries: int = 3, retry_delay: float = 1.0):
        parser = PydanticOutputParser(pydantic_object=output_cls)

        async def ainvoke(messages: list[BaseMessage], **kwargs) -> tuple[T, dict]:
            format_instructions = parser.get_format_instructions()
            augmented = [
                SystemMessage(content=f"You must respond with valid JSON.\n{format_instructions}"),
                *messages,
            ]

            last_error = None
            for attempt in range(max_retries):
                try:
                    response = await self.llm.ainvoke(augmented, **kwargs)
                    usage = self._extract_usage(response)
                    from typing import cast
                    raw = (
                        cast(str, response.content)
                        .strip()
                        .removeprefix("```json")
                        .removeprefix("```")
                        .removesuffix("```")
                        .strip()
                    )
                    repaired = json.loads(raw)
                    parsed = parser.parse(json.dumps(repaired))
                    return parsed, usage
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"[OpenRouter] Structured LLM attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {retry_delay}s..."
                        )
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(
                            f"[OpenRouter] Structured LLM failed after {max_retries} attempts: {e}"
                        )
            raise last_error if last_error else RuntimeError("Structured LLM failed with unknown error")

        return ainvoke

    async def achat(
        self,
        messages: list[BaseMessage],
        **kwargs,
    ) -> AIMessage:
        """
        Send a chat completion using LangChain message format.

        Args:
            messages: List of LangChain BaseMessage objects
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            AIMessage response
        """
        response = await self.llm.ainvoke(messages, **kwargs)
        return response  # type: ignore[return-value]

    def _extract_usage(self, response: AIMessage) -> dict:
        """Extract usage info from LangChain AIMessage."""
        token_usage = response.response_metadata.get('token_usage', {})
        completion_tokens = token_usage.get('completion_tokens', 0)
        prompt_tokens = token_usage.get('prompt_tokens', 0)
        total_tokens = token_usage.get('total_tokens', 0)
        
        
        cost = token_usage.get('cost', 0.0)
        return {
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
        }

    def chat(
        self,
        messages: list[BaseMessage],
        **kwargs,
    ) -> AIMessage:
        """
        Synchronous chat completion.

        Args:
            messages: List of LangChain BaseMessage objects
            **kwargs: Additional parameters

        Returns:
            AIMessage response
        """
        response = self.llm.invoke(messages, **kwargs)
        return response  # type: ignore[return-value]

    async def acomplete(
        self,
        prompt: str,
        **kwargs,
    ) -> AIMessage:
        """
        Text completion interface (wraps a single HumanMessage).

        Args:
            prompt: Text prompt for completion
            **kwargs: Additional parameters

        Returns:
            AIMessage with completion
        """
        return await self.achat([HumanMessage(content=prompt)], **kwargs)

    def complete(
        self,
        prompt: str,
        **kwargs,
    ) -> AIMessage:
        """
        Synchronous text completion.

        Args:
            prompt: Text prompt for completion
            **kwargs: Additional parameters

        Returns:
            AIMessage with completion
        """
        return self.chat([HumanMessage(content=prompt)], **kwargs)

    def stream_chat(
        self,
        messages: list[BaseMessage],
        **kwargs,
    ):
        """
        Streaming chat completion.

        Args:
            messages: List of LangChain BaseMessage objects
            **kwargs: Additional parameters

        Yields:
            AIMessageChunk deltas
        """
        yield from self.llm.stream(messages, **kwargs)

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
            AIMessageChunk deltas
        """
        yield from self.stream_chat([HumanMessage(content=prompt)], **kwargs)

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

    @staticmethod
    def _prepare_image_url(image_data: bytes | str) -> tuple[str, str]:
        """
        Normalize image input to a (data_url, mime_type) pair.

        LangChain's multimodal HumanMessage expects an image_url dict:
          {"url": "data:<mime>;base64,<b64>"}
        """
        if isinstance(image_data, str) and image_data.startswith("data:"):
            mime = image_data.split(":")[1].split(";")[0]
            return image_data, mime
        elif isinstance(image_data, bytes):
            b64 = base64.b64encode(image_data).decode("utf-8")
            mime = "image/jpeg"
            return f"data:{mime};base64,{b64}", mime
        else:
            # Assume file path
            path = Path(image_data)  # type: ignore[arg-type]
            ext = path.suffix.lower().lstrip(".")
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
            b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
            return f"data:{mime};base64,{b64}", mime

    async def ainfer_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> OpenRouterResult | None:
        """
        Text-only inference.

        Args:
            prompt: User message
            system_prompt: Optional system message
            **kwargs: Additional parameters

        Returns:
            OpenRouterResult or None on error
        """
        try:
            messages: list[BaseMessage] = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            response = await self.llm.ainvoke(messages, **kwargs)
            usage = self._extract_usage(response)  # type: ignore[arg-type]

            return OpenRouterResult(
                content=response.content if isinstance(response.content, str) else str(response.content),
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
        Image (+ optional text) inference.

        LangChain multimodal messages use a content list with type=image_url dicts.

        Args:
            image_data: Image bytes or data URI string
            prompt: Text accompanying the image
            system_prompt: Optional system message
            **kwargs: Additional parameters

        Returns:
            OpenRouterResult or None on error
        """
        try:
            messages: list[BaseMessage] = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            data_url, _ = self._prepare_image_url(image_data)
            messages.append(
                HumanMessage(
                    content=[
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ]
                )
            )

            response = await self.llm.ainvoke(messages, **kwargs)
            usage = self._extract_usage(response)  # type: ignore[arg-type]

            return OpenRouterResult(
                content=response.content if isinstance(response.content, str) else str(response.content),
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
        Multi-image inference.

        Args:
            image_data_list: List of image bytes or data URIs
            prompt: Text accompanying the images
            system_prompt: Optional system message
            **kwargs: Additional parameters

        Returns:
            OpenRouterResult or None on error
        """
        try:
            messages: list[BaseMessage] = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            content: list[dict] = []
            for image_data in image_data_list:
                data_url, _ = self._prepare_image_url(image_data)
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            content.append({"type": "text", "text": prompt})

            human_message = HumanMessage(content=content) #type:ignore
            messages.append(human_message)

            response = await self.llm.ainvoke(messages, **kwargs)
            usage = self._extract_usage(response)  # type: ignore[arg-type]

            return OpenRouterResult(
                content=response.content if isinstance(response.content, str) else str(response.content),
                model=self.config.model,
                finish_reason="stop",
                usage=usage,
            )
        except Exception as e:
            logger.exception(f"Multi-image inference failed: {e}")
            return None

    async def ainfer_chat(
        self,
        messages: list[BaseMessage],
        **kwargs,
    ) -> OpenRouterResult | None:
        """
        Chat completion using LangChain BaseMessage format.

        Args:
            messages: List of LangChain BaseMessage objects
            **kwargs: Additional parameters

        Returns:
            OpenRouterResult or None on error
        """
        try:
            response = await self.llm.ainvoke(messages, **kwargs)
            usage = self._extract_usage(response)  # type: ignore[arg-type]

            return OpenRouterResult(
                content=response.content if isinstance(response.content, str) else str(response.content),
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
        messages_list: list[list[BaseMessage]],
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

        async def _limited(messages: list[BaseMessage]):
            async with semaphore:
                return await self.ainfer_chat(messages, **kwargs)

        return list(await asyncio.gather(*[_limited(msgs) for msgs in messages_list]))