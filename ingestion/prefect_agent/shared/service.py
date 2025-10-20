
from __future__ import annotations

import time
from abc import ABC
from typing import Any, Generic, Literal, Optional, TypeVar, cast

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

import asyncio
import os
from shared.config import LogConfig, ServiceConfig
from shared.logger import setup_service_logger
from shared.metrics import ServiceMetrics
from shared.monitor import (
    get_cpu_usage,
    get_gpu_memory_info,
    get_gpu_name,
    get_gpu_utilization,
    get_ram_info,
    is_gpu_available,
)
from shared.registry import BaseModelHandler, get_model_handler, list_models
from shared.schema import ModelInfo

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseService(Generic[InputT, OutputT], ABC):

    def __init__(self, service_config: ServiceConfig, log_config: LogConfig) -> None:
        self.service_config = service_config

        setup_service_logger(
            service_name=service_config.service_name,
            service_version=service_config.service_version,
            log_level=log_config.log_level,
            log_format=log_config.log_format,
            log_file=log_config.log_file,
            log_rotation=log_config.log_rotation,
            log_retention=log_config.log_retention,
        )

        self.loaded_model: Optional[BaseModelHandler[InputT, OutputT]] = None
        self.loaded_model_info: Optional[ModelInfo] = None
        self.current_device: Optional[str] = None

        self.metrics = ServiceMetrics(service_name=service_config.service_name)
        

        self.active_request = 0

        logger.info(
            "service_initialized",
            service=service_config.service_name,
            gpu_available=is_gpu_available(),
        )

    def get_available_models(self) -> list[str]:
        return list_models()

    def update_system_metrics(self) -> None:
        cpu_percent = get_cpu_usage()
        self.metrics.update_cpu_usage(cpu_percent)

        ram_info = get_ram_info()
        used_bytes = int(ram_info["used_mb"] * 1024 * 1024)
        self.metrics.update_memory_usage(used_bytes, percent_used=ram_info["percent"])

        if is_gpu_available():
            gpu_info = get_gpu_memory_info()
            if gpu_info:
                gpu_mem = cast(dict[str, float], get_gpu_memory_info())
                gpu_util = get_gpu_utilization()

                self.metrics.update_gpu_metrics(
                    gpu_id=0,
                    memory_used=int(gpu_mem["allocated_mb"] * 1024 * 1024),
                    memory_total=int(gpu_mem["total_mb"] * 1024 * 1024),
                    utilization=gpu_util,
                )

    async def load_model(
        self,
        model_name: str,
        device: Literal["cuda", "cpu"],
    ) -> ModelInfo:
        if model_name not in self.get_available_models():
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' not available. Available: {self.get_available_models()}",
            )
        if device == "cuda" and not is_gpu_available():
            logger.warning(
                "gpu_not_available_fallback_to_cpu",
                requested_device=device,
            )
            if self.service_config.cpu_fallback:
                device = "cpu"
            else:
                raise HTTPException(
                    status_code=503,
                    detail="GPU not available and CPU fallback disabled",
                )


        if self.loaded_model is not None:
            current_registry_name = getattr(self.loaded_model, "model_name", None)
            if current_registry_name == model_name:
                logger.info(
                    "model_already_loaded",
                    model=model_name,
                    device=self.current_device,
                )
                self.update_system_metrics()
                return cast(ModelInfo, self.loaded_model_info)

        logger.info("loading_model", model=model_name, device=device)

        try:
            handler = cast(
                BaseModelHandler[InputT, OutputT],
                get_model_handler(model_name, self.service_config),
            )
            await handler.load_model_impl(device)
            self.loaded_model = handler
            self.loaded_model_info = handler.get_model_info()
            self.current_device = device

            if device == "cuda":
                mem_info = cast(dict[str, float], get_gpu_memory_info())
                logger.info(
                    "model_loaded_gpu",
                    model=model_name,
                    allocated_mb=mem_info["allocated_mb"],
                    free_mb=mem_info["free_mb"],
                )
            logger.info("model_loaded_success", model=model_name, device=device)
            self.update_system_metrics()

            return self.loaded_model_info
        except Exception as exc: 

            logger.exception(
                "model_load_failed",
                model=model_name,
                device=device,
                error=str(exc),
            )
            self.metrics.track_error("load_model", type(exc).__name__)
            raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}")

    async def unload_model(self, cleanup_memory: bool = True) -> None:
        if self.loaded_model is None:
            logger.warning("no_model_loaded")
            return

        model_name = self.loaded_model_info.model_name if self.loaded_model_info else "<unknown>"
        device = self.current_device

        logger.info("unloading_model", model=model_name, device=device)

        try:
            if self.active_request > 0:
                logger.info(f"Some client still using this, no unloading...")
                return

            await self.loaded_model.unload_model_impl()
            self.loaded_model = None
            self.loaded_model_info = None
            self.current_device = None

            if cleanup_memory and device == "cuda" and is_gpu_available():
                torch = __import__("torch")
                torch.cuda.empty_cache()
                mem_info = cast(dict[str, float], get_gpu_memory_info())
                logger.info(
                    "gpu_memory_after_cleanup",
                    free_mb=mem_info["free_mb"],
                )
            logger.info("model_unloaded", model=model_name)
            self.update_system_metrics()
        except Exception as exc:  
            logger.exception("model_unload_failed", error=str(exc))
            raise HTTPException(status_code=500, detail=f"Failed to unload model: {exc}")

    async def infer(self, input_data: InputT) -> OutputT:
        if self.loaded_model is None:
            logger.error(
                "no_model_initialized",
                service=self.service_config.service_name,
            )
            raise HTTPException(status_code=400, detail="No model loaded. Load a model before inference.")

        try:
            start_time = time.time()
            metadata = getattr(input_data, "metadata", {})

            logger.info("********************************************")
            logger.info("preprocessing_input", metadata=metadata)
            handler = self.loaded_model
            self.active_request += 1
            preprocessed = await handler.preprocess_input(input_data)  

            logger.info("running_inference")
            result = await handler.run_inference(preprocessed)  

            logger.info("postprocessing_output")
            output = await handler.postprocess_output(result, input_data)  

            duration = time.time() - start_time
            self.metrics.observe_request_duration("infer", duration)
            self.metrics.track_request("infer", "success")

            logger.info(
                "inference_complete",
                duration_seconds=duration,
                model=self.loaded_model_info.model_name if self.loaded_model_info else None,
            )
            logger.info("********************************************")

            return output
        except Exception as exc: 
            logger.exception("inference_failed", error=str(exc))
            self.metrics.track_error("infer", type(exc).__name__)
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
        finally:
            try:
                if self.active_request > 0:
                    self.active_request -= 1
            except Exception:
                pass

    def get_system_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "cpu_percent": get_cpu_usage(),
            "ram": get_ram_info(),
            "model_loaded": self.loaded_model is not None,
            "model_info": self.loaded_model_info.model_dump(mode="json") if self.loaded_model_info else None,
        }

        if is_gpu_available():
            status["gpu"] = {
                "available": True,
                "name": get_gpu_name(),
                "memory": get_gpu_memory_info(),
                "utilization": get_gpu_utilization(),
            }
        else:
            status["gpu"] = {"available": False}

        return status
