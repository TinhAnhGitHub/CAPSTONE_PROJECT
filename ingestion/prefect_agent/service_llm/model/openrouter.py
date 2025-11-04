from __future__ import annotations
import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Dict, Literal

import requests
from loguru import logger
from PIL import Image
from service_llm.core.config import LLMServiceConfig
from service_llm.schema import LLMRequest, LLMResponse,LLMSingleResponse
from shared.registry import BaseModelHandler, register_model
from shared.schema import ModelInfo


# def _encode_image(path: str) -> str:
#     with Image.open(path) as img:
#         if img.mode != "RGB":
#             img = img.convert("RGB")
#         with Path(path).open("rb") as fh:
#             return base64.b64encode(fh.read()).decode("utf-8")


MAX_CURRENT = 10

@register_model("openrouter_api")
class OpenRouterAPIHandler(BaseModelHandler[LLMRequest, LLMResponse]):
    """Handler that routes multimodal prompts to OpenRouter-compatible endpoints."""

    def __init__(self, model_name: str, config: LLMServiceConfig) -> None:
        super().__init__(model_name, config)
        if not config.openrouter_api_key:
            raise ValueError("OpenRouter handler requires OPENROUTER_API_KEY to be set")
        self._api_key = config.openrouter_api_key
        self._model_name = config.openrouter_model_name
        self._endpoint = config.openrouter_base_url
        self._referer = config.openrouter_referer
        self._title = config.openrouter_title or "Capstone LLM Service"
        self._session: requests.Session = None #type:ignore


    async def load_model_impl(self, device: Literal["cpu", "cuda"]) -> None:  # noqa: ARG002 device unused
        if self._session is not None:
            return
        self._session = requests.Session()
        logger.info("openrouter_client_initialized", endpoint=self._endpoint)

    async def unload_model_impl(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None #type:ignore

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_name=self._model_name, model_type="openrouter_api")

    async def preprocess_input(self, input_data: LLMRequest) -> list[Dict[str, Any]]:

        return [{
            "prompt": req.prompt,
            "image_base64": req.image_base64,
            "metadata": input_data.metadata,
        } for req  in input_data.llm_requests]
    

    async def _run_inference(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        if self._session is None:
            raise RuntimeError("OpenRouter session not initialized")

        prompt = preprocessed_data["prompt"]
        image_base64 = preprocessed_data["image_base64"]

        def _call_api() -> Dict[str, Any]:
            content: list[Dict[str, Any]] = []
            if prompt:
                content.append({"type": "text", "text": prompt})
            
            if image_base64:
                for img_b64 in image_base64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })

            payload = {
                "model": self._model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "max_tokens": None,
            }

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "X-Title": self._title,
            }
            if self._referer:
                headers["HTTP-Referer"] = self._referer

            response = self._session.post(self._endpoint, data=json.dumps(payload), headers=headers, timeout=60)
            if response.status_code >= 400:
                logger.error(
                    "openrouter_request_failed",
                    status=response.status_code,
                    body=response.text,
                )
                response.raise_for_status()
        
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return {"answer": "", "input_tokens": None, "output_tokens": None}

            message = choices[0].get("message", {})
            message_content = message.get("content", "")
            if isinstance(message_content, list):
                answer_text = "".join(part.get("text", "") for part in message_content)
            else:
                answer_text = str(message_content)

            usage = data.get("usage", {}) or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")

            return {
                "answer": answer_text,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
            }

        response_payload = await asyncio.to_thread(_call_api)
        return response_payload
    
    async def run_inference(self, preprocessed_data: list[Dict[str, Any]]) -> list[Dict[str, Any]]:        
        semaphore = asyncio.Semaphore(MAX_CURRENT)

        async def _infer_with_semaphore(data):
            async with semaphore:
                return await self._run_inference(preprocessed_data=data)
        
        inference_tasks = [
            asyncio.create_task(_infer_with_semaphore(p)) for p in preprocessed_data
        ]
        inference_results = await asyncio.gather(*inference_tasks, return_exceptions=True)
        results: list[Dict[str, Any]] = []
        for r in inference_results:
            if isinstance(r, BaseException):
                raise RuntimeError("OpenRouter request failed") from r
            else:
                results.append(r)
        return results
    

    async def postprocess_output(
        self,
        output_data: list[Dict[str, Any]],
        original_input_data: LLMRequest,
    ) -> LLMResponse:
        
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
