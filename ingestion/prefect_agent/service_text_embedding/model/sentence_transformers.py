from __future__ import annotations

from typing import Any, Dict, Literal
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger

from service_text_embedding.core.config import TextEmbeddingConfig
from service_text_embedding.schema import TextEmbeddingRequest, TextEmbeddingResponse
from shared.registry import BaseModelHandler, register_model
from shared.schema import ModelInfo


@register_model("sentence_embedding")
class SentenceTransformerHandler(BaseModelHandler[TextEmbeddingRequest, TextEmbeddingResponse]):
    """Model handler built on top of SentenceTransformer backbones."""

    def __init__(self, model_name: str, config: TextEmbeddingConfig) -> None:
        super().__init__(model_name, config)
        self._service_config = config
        self._model_id = config.sentence_transformer_model
        self._model: SentenceTransformer | None = None
        self._device: str | None = None

    async def load_model_impl(self, device: Literal["cpu", "cuda"]) -> None:
        if self._model is not None:
            return

        actual_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(self._model_id, device=actual_device)
        self._device = actual_device

    async def unload_model_impl(self) -> None:
        if self._model is None:
            return
        del self._model
        self._model = None
        self._device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_name=self._model_id, model_type="text_embedding")

    async def preprocess_input(self, input_data: TextEmbeddingRequest) -> Dict[str, Any]:
        if not input_data.texts:
            raise ValueError("No texts provided for embedding")
        batch_size = self._service_config.sentence_transformer_batch_size
        return {
            "texts": input_data.texts,
            "batch_size": batch_size,
            "metadata": input_data.metadata,
        }

    async def run_inference(self, preprocessed_data: Dict[str, Any]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("SentenceTransformer model not loaded")

        embeddings = self._model.encode(  # type: ignore[union-attr]
            preprocessed_data["texts"],
            batch_size=preprocessed_data["batch_size"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return embeddings

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
