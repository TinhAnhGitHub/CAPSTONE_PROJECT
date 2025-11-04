from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Literal

from loguru import logger
from PIL import Image

from service_llm.core.config import LLMServiceConfig
from service_llm.schema import LLMRequest, LLMResponse, LLMSingleResponse
from shared.registry import BaseModelHandler, register_model
from shared.schema import ModelInfo
import base64
import io



MAX_CURRENT = 10


def _load_image_from_b64(b64_str: str) -> Image.Image:
    """Decode a base64 image string into a PIL image."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    img_data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

@register_model("gemini_api")
class GeminiAPIHandler(BaseModelHandler[LLMRequest, LLMResponse]):

    def __init__(self, model_name: str, config: LLMServiceConfig) -> None:
        super().__init__(model_name, config)
        if not config.gemini_api_key:
            raise ValueError("Gemini API handler requires GEMINI_API_KEY to be set")
        self._api_key = config.gemini_api_key
        self._model_name = config.gemini_model_name
        self._client = None

    async def load_model_impl(self, device: Literal["cpu", "cuda"]) -> None:  
        if self._client is not None:
            return
        try:
            import google.generativeai as genai
        except ImportError as exc: 
            raise RuntimeError(
                "google-generativeai package is required for Gemini handler"
            ) from exc

        genai.configure(api_key=self._api_key) # type: ignore[attr-defined]
        self._client = genai.GenerativeModel(self._model_name) # type: ignore[attr-defined]
        logger.info("gemini_client_initialized", model=self._model_name)

    async def unload_model_impl(self) -> None:
        self._client = None

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_name=self._model_name, model_type="gemini_api")

    async def preprocess_input(self, input_data: LLMRequest) -> list[Dict[str, Any]]:
        return [
            {
                "prompt": req.prompt,
                "image_base64": req.image_base64,
                "metadata": input_data.metadata,
            }
            for req in input_data.llm_requests
        ]

    async def _run_inference(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        if self._client is None:
            raise RuntimeError("Gemini client not initialized")

        prompt = preprocessed_data.get("prompt")
        image_b64_list = preprocessed_data.get("image_base64", [])

        def _call_api() -> Dict[str, Any]:
            try:
                parts: list[Any] = []
                for b64_str in image_b64_list or []:
                    parts.append(_load_image_from_b64(b64_str))
                if prompt:
                    parts.append(prompt)

                response = self._client.generate_content(parts)  # type: ignore[union-attr]

                answer_text = getattr(response, "text", "") or ""
                usage_metadata = getattr(response, "usage_metadata", None)
                prompt_tokens = (
                    getattr(usage_metadata, "prompt_token_count", None)
                    if usage_metadata
                    else None
                )
                completion_tokens = (
                    getattr(usage_metadata, "candidates_token_count", None)
                    if usage_metadata
                    else None
                )

                return {
                    "answer": answer_text,
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                }

            except Exception as e:
                logger.error(f"Gemini inference error: {e}")
                return {"answer": "", "error": str(e)}

        return await asyncio.to_thread(_call_api)
    

    async def run_inference(self, preprocessed_data: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Concurrent execution with semaphore for multiple Gemini calls."""
        semaphore = asyncio.Semaphore(MAX_CURRENT)

        async def _infer_with_semaphore(data):
            async with semaphore:
                return await self._run_inference(preprocessed_data=data)

        tasks = [asyncio.create_task(_infer_with_semaphore(p)) for p in preprocessed_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: list[Dict[str, Any]] = []
        for r in results:
            if isinstance(r, BaseException):
                raise RuntimeError("OpenRouter request failed") from r
            else:
                output.append(r)
        return output

    async def postprocess_output(
        self, output_data: list[Dict[str, Any]], original_input_data: LLMRequest
    ) -> LLMResponse:
        """Convert raw Gemini outputs into standardized LLMResponse."""
        single_responses: list[LLMSingleResponse] = []
        for out in output_data:
            single_responses.append(
                LLMSingleResponse(
                    answer=out.get("answer", ""),
                    input_tokens=out.get("input_tokens"),
                    output_tokens=out.get("output_tokens"),
                    status="success" if "error" not in out else "error",
                    error=out.get("error"),
                )
            )

        overall_status = (
            "success"
            if all(r.status == "success" for r in single_responses)
            else "partial_error"
        )

        return LLMResponse(
            responses=single_responses,
            metadata=original_input_data.metadata,
            model_name=self._model_name,
            status=overall_status,
        )