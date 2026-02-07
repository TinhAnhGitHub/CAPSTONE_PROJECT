"""
Docstring for video_pipeline.core.client.autoshot_client
1. Triton compatible
2. Using async to call
"""

import numpy as np
import tritonclient.grpc as grpcclient
import asyncio
from loguru import logger
from pydantic import BaseModel


class TritonConfig(BaseModel):
    model_name: str
    model_version: str


class AutoShotClient:
    def __init__(self, client_url: str, triton_config: TritonConfig):
        self.client_url = client_url
        self.model_name = triton_config.model_name
        self.model_version = triton_config.model_version
        self.client = grpcclient.InferenceServerClient(url=client_url)

    def check_server_health(self) -> bool:
        """Check if Triton server is alive and model is ready"""
        try:
            if not self.client.is_server_live():
                logger.error("Triton server is not live")
                return False

            if not self.client.is_server_ready():
                logger.error("Triton server is not ready")
                return False

            if not self.client.is_model_ready(self.model_name, self.model_version):
                logger.error(
                    f"Model {self.model_name} version {self.model_version} is not ready"
                )
                return False

            logger.success(f"Server and model {self.model_name} are ready")
            return True
        except Exception as e:
            logger.error(f"Error checking server health: {e}")
            return False

    async def ainfer(self, frames: np.ndarray) -> np.ndarray | None:
        """
        :param frames: np.ndarray of shape (1, 100, 27, 48, 3), dtype=uint8
        """

        loop = asyncio.get_running_loop()
        future: asyncio.Future[np.ndarray] = loop.create_future()

        def callback(result, error):
            if error is not None:
                loop.call_soon_threadsafe(future.set_exception, error)
            else:
                try:
                    output = result.as_numpy("shot_boundary_prob")
                    loop.call_soon_threadsafe(future.set_result, output)
                except Exception as e:
                    loop.call_soon_threadsafe(future.set_exception, e)

        try:
            inputs = [
                grpcclient.InferInput(
                    name="input_frames", shape=frames.shape, datatype="UINT8"
                )
            ]
            inputs[0].set_data_from_numpy(frames)

            outputs = [grpcclient.InferRequestedOutput("shot_boundary_prob")]

            self.client.async_infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=inputs,
                outputs=outputs,
                callback=callback,
            )

            return await future

        except Exception:
            logger.exception("Async inference failed")
            return None

    async def batch_async_infer(self, batch_frames: list) -> list:
        """
        Run multiple inferences concurrently

        Args:
            batch_frames: List of frame arrays, each of shape (1, 100, 27, 48, 3)

        Returns:
            List of shot boundary probabilities
        """
        tasks = [self.ainfer(frames) for frames in batch_frames]
        results = await asyncio.gather(*tasks)
        return results

    def close(self):
        self.client.close()

    def load_model(self) -> bool:
        try:
            self.client.load_model(model_name=self.model_name, config=None)
            logger.success(f"Model {self.model_name} loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def unload_model(self) -> bool:
        try:
            self.client.unload_model(
                model_name=self.model_name,
            )
            logger.success(f"Model {self.model_name} unloaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return False

    def get_model_metadata(self) -> dict | None:
        try:
            metadata = self.client.get_model_metadata(
                model_name=self.model_name, model_version=self.model_version
            )
            return metadata
        except Exception as e:
            logger.error(f"Error getting model metadata: {e}")
            return None
