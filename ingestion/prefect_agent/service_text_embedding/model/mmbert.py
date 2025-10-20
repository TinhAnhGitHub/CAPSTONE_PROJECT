from __future__ import annotations

from typing import Any, Dict, Literal
from pathlib import Path

import numpy as np
import torch #type:ignore
from transformers import AutoModel, AutoTokenizer
from loguru import logger

from service_text_embedding.core.config import TextEmbeddingConfig
from service_text_embedding.schema import TextEmbeddingRequest, TextEmbeddingResponse
from shared.registry import BaseModelHandler, register_model
from shared.schema import ModelInfo
torch.set_float32_matmul_precision('high')

@register_model("mmbert")
class MMBERTHandler(BaseModelHandler[TextEmbeddingRequest, TextEmbeddingResponse]):
    """mmBERT-based text embedding handler."""

    def __init__(self, model_name: str, config: TextEmbeddingConfig) -> None:
        super().__init__(model_name, config)
        if not config.mmbert_model_name:
            raise ValueError("MMBERT model requested but 'mmbert_model_name' is not configured")
        self._service_config = config
        self._checkpoint = config.mmbert_model_name
        self._max_length = config.mmbert_max_length
        self._model: AutoModel | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._device: str | None = None

    async def load_model_impl(self, device: Literal["cpu", "cuda"]) -> None:
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self._checkpoint)
        model = AutoModel.from_pretrained(self._checkpoint)

        actual_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self._model = model.to(actual_device)
        self._model.eval() #type:ignore
        self._device = actual_device

    async def unload_model_impl(self) -> None:
        if self._model is None:
            return
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_name=self._checkpoint, model_type="text_embedding")

    async def preprocess_input(self, input_data: TextEmbeddingRequest) -> Dict[str, Any]:
        if self._tokenizer is None:
            raise RuntimeError("mmBERT tokenizer not loaded")

        if not input_data.texts:
            raise ValueError("No texts provided for embedding")

        batch_size = self._service_config.mmbert_batch_size
        return {
            "texts": input_data.texts,
            "batch_size": batch_size,
            "metadata": input_data.metadata,
        }

    async def run_inference(self, preprocessed_data: Dict[str, Any]) -> np.ndarray:
        if self._model is None or self._tokenizer is None or self._device is None:
            raise RuntimeError("mmBERT model not loaded")

        texts = preprocessed_data["texts"]
        batch_size = preprocessed_data["batch_size"]
        all_embeddings = []

        for idx in range(0, len(texts), batch_size):
            chunk = texts[idx : idx + batch_size]
            inputs = self._tokenizer( #type:ignore
                chunk,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)#type:ignore
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = embeddings.cpu().numpy().astype(np.float32)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    async def postprocess_output(
        self,
        output_data: np.ndarray,
        original_input_data: TextEmbeddingRequest,
    ) -> TextEmbeddingResponse:
        return TextEmbeddingResponse(
            embeddings=output_data.tolist(),
            texts=original_input_data.texts,
            metadata=original_input_data.metadata,
            status="success",
        )
