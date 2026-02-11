import asyncio
import json
import numpy as np
from loguru import logger
import tritonclient.grpc as grpcclient
from pydantic import BaseModel


class ParakeetASRConfig(BaseModel):
    model_name: str = "parakeet_asr"
    model_version: str = "1"


class ParakeetASRClient:
    def __init__(self, client_url: str, triton_config: ParakeetASRConfig):
        self.client_url = client_url
        self.model_name = triton_config.model_name
        self.model_version = triton_config.model_version
        self.client = grpcclient.InferenceServerClient(url=client_url)

    def check_server_health(self) -> bool:
        try:
            if not self.client.is_server_live():
                logger.error("Triton server is not live")
                return False
            if not self.client.is_server_ready():
                logger.error("Triton server is not ready")
                return False
            if not self.client.is_model_ready(self.model_name, self.model_version):
                logger.error("ASR model not ready")
                return False
            logger.success("ASR Triton model is ready")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def ainfer(self, audio_paths: list[str]) -> list[str] | None:
        """
        audio_paths: list of audio file paths
        """

        loop = asyncio.get_running_loop()
        future: asyncio.Future[np.ndarray] = loop.create_future()

        def callback(result, error):
            if error:
                loop.call_soon_threadsafe(future.set_exception, error)
            else:
                try:
                    output = result.as_numpy("OUTPUT")
                    loop.call_soon_threadsafe(future.set_result, output)
                except Exception as e:
                    loop.call_soon_threadsafe(future.set_exception, e)

        try:
            audio_bytes_list = []
            for p in audio_paths:
                with open(p, "rb") as f:
                    audio_bytes_list.append([f.read()])
            audio_np = np.array(audio_bytes_list, dtype=object)

            inputs = [
                grpcclient.InferInput(
                    name="AUDIO",
                    shape=audio_np.shape,
                    datatype="BYTES",
                )
            ]
            inputs[0].set_data_from_numpy(audio_np)

            outputs = [grpcclient.InferRequestedOutput("OUTPUT")]

            self.client.async_infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=inputs,
                outputs=outputs,
                callback=callback,
            )

            result = await future

            # Decode UTF-8 strings

            final_output = []
            for item in result.flatten():
                if isinstance(item, bytes):
                    json_str = item.decode("utf-8")
                    data = json.loads(json_str)
                    texts = [entry.get("text", "") for entry in data]
                    final_output.extend(texts)

            return final_output
        except Exception:
            logger.exception("ASR async inference failed")
            return None

    def close(self):
        self.client.close()
