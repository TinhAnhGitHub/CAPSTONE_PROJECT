from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

from loguru import logger

from service_asr.model.asr_core import ASRProcessor
from service_asr.core.config import ASRServiceConfig
from service_asr.core.schema import ASRConfig, ASRInferenceRequest, ASRInferenceResponse, ASRResult
from shared.registry import BaseModelHandler, register_model
from shared.schema import ModelInfo
from service_asr.util import resolve_video_url
from shared.storage import MinioSettings, StorageClient
from shared.util import fetch_object_from_s3


@dataclass(slots=True)
class _PreparedInput:
    video_path: Path
    audio_path: Path
    config: ASRConfig
    delete_video_on_cleanup: bool


@register_model("chunkformer")
class ChunkformerASRHandler(BaseModelHandler[ASRInferenceRequest, ASRInferenceResponse]):
    """Model handler that wraps the Chunkformer-based ASRProcessor."""

    def __init__(self, model_name: str, config: ASRServiceConfig) -> None:
        super().__init__(model_name, config)
        self._service_config = config
        self._processor: ASRProcessor | None = None
        self._current_device: str | None = None

    async def load_model_impl(self, device: Literal["cpu", "cuda"]) -> None:
        if self._processor is not None:
            return
        logger.info("loading_chunkformer_model", path=self._service_config.model_name, device=device)
        self._processor = ASRProcessor(
            model_name=self._service_config.model_name,
            device=device,
        )
        self._current_device = device

    async def unload_model_impl(self) -> None:
        if self._processor is None:
            return
        logger.info("unloading_chunkformer_model")
        if self._current_device == "cuda":
            import torch

            torch.cuda.empty_cache()
        self._processor = None
        self._current_device = None

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_name="chunkformer", model_type="asr")

    async def preprocess_input(self, input_data: ASRInferenceRequest) -> _PreparedInput:
        if self._processor is None:
            raise RuntimeError("Chunkformer model not loaded")

        config = input_data.config or self._default_config()

        source = input_data.video_minio_url or getattr(input_data, "video_path", None)
        if not source:
            raise ValueError("No input video provided (expected 'video_minio_url' or 'video_path')")

        delete_video_on_cleanup = False
        local_video_path: Path
        logger.info("fetching_video_from_s3", url=source)
        storage = StorageClient(MinioSettings())
        local_path_str = await fetch_object_from_s3(source, storage, suffix=".mp4")
        local_video_path = Path(local_path_str)
        delete_video_on_cleanup = True

        if not local_video_path.exists():
            raise FileNotFoundError(f"Video file not found: {local_video_path}")

        temp_dir = Path(self._service_config.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        audio_path = temp_dir / f"{uuid.uuid4().hex}"

        logger.info("extracting_audio", video=str(local_video_path))
        success = await asyncio.to_thread(
            self._processor.extract_audio,
            local_video_path,
            audio_path,
            config.sample_rate,
        )
        if not success:
            if delete_video_on_cleanup and local_video_path.exists():
                try:
                    local_video_path.unlink()
                except OSError:
                    logger.warning("video_cleanup_failed_on_error", path=str(local_video_path))
            raise RuntimeError("Audio extraction failed")

        return _PreparedInput(
            video_path=local_video_path,
            audio_path=audio_path.with_suffix(".wav"),
            config=config,
            delete_video_on_cleanup=delete_video_on_cleanup,
        )

    async def run_inference(self, preprocessed_data: _PreparedInput) -> Dict[str, Any]:
        if self._processor is None:
            raise RuntimeError("Chunkformer model not loaded")

        logger.info("running_asr_inference", video=str(preprocessed_data.video_path))
        try:
            result: ASRResult = await asyncio.to_thread(
                self._processor.process_audio,
                preprocessed_data.audio_path,
                str(preprocessed_data.video_path),
                preprocessed_data.config,
            )
        except Exception:
            if preprocessed_data.audio_path.exists():
                try:
                    preprocessed_data.audio_path.unlink()
                except OSError:
                    logger.warning("audio_cleanup_failed_on_error", path=str(preprocessed_data.audio_path))
            raise
        return {
            "result": result,
            "audio_path": preprocessed_data.audio_path,
            "video_path": preprocessed_data.video_path,
            "delete_video_on_cleanup": preprocessed_data.delete_video_on_cleanup,
        }

    async def postprocess_output(
        self,
        output_data: Dict[str, Any],
        original_input_data: ASRInferenceRequest,
    ) -> ASRInferenceResponse:
        audio_path: Path = output_data["audio_path"]
        if audio_path.exists():
            try:
                audio_path.unlink()
            except OSError:
                logger.warning("audio_cleanup_failed", path=str(audio_path))

        # Cleanup transient video file if we created one
        try:
            if output_data.get("delete_video_on_cleanup"):
                video_path: Path = output_data["video_path"]
                if video_path.exists():
                    video_path.unlink()
        except Exception:
            logger.warning("video_cleanup_failed", path=str(output_data.get("video_path")))

        result: ASRResult = output_data["result"]
        # Preserve original source if provided; otherwise return the local path
        source = original_input_data.video_minio_url or getattr(original_input_data, "video_path", str(output_data.get("video_path")))
        return ASRInferenceResponse(
            video_minio_url=source,
            metadata=original_input_data.metadata,
            result=result,
            status="success",
        )

    def _default_config(self) -> ASRConfig:
        cfg = self._service_config
        return ASRConfig(
            chunk_size=cfg.default_chunk_size,
            left_context_size=cfg.default_left_context,
            right_context_size=cfg.default_right_context,
            total_batch_duration=cfg.default_total_batch_duration,
            sample_rate=cfg.default_sample_rate,
            num_extraction_workers=cfg.default_num_extraction_workers,
            num_asr_workers=cfg.default_num_asr_workers,
        )
