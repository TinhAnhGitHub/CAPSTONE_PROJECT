import triton_python_backend_utils as pb_utils #type:ignore
import numpy as np
import json
import tempfile
import os
import uuid
from pathlib import Path
import base64
from typing import List, Dict, Any
import sys

from asr_core import ASRProcessor
from schema import ASRConfig




class TritonPythonModel:
    def initialize(self, args):
        """Called once when model is loaded."""
        self.model_config = json.loads(args["model_config"])
        self.device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        
        parameters = self.model_config.get("parameters", {})


        print(f"ASR parameters: {parameters=}")
        self.model_name = parameters["model_name"]["string_value"]
        self.chunk_size = int(parameters["chunk_size"]["string_value"])
        self.left_context = int(parameters["left_context_size"]["string_value"])
        self.right_context = int(parameters["right_context_size"]["string_value"])
                

        self.processor = ASRProcessor(
            model_name=self.model_name,
            device=self.device
        )
        
     
        self.temp_dir = Path("/tmp/asr_temp")
        self.temp_dir.mkdir(exist_ok=True)

    def execute(self, requests: list) -> list:
        """Called for each batch of requests."""
        responses = []
        
        for request in requests:
            try:
                
                video_input = pb_utils.get_input_tensor_by_name(request, "video_input")
                video_bytes = video_input.as_numpy()[0][0].decode("utf-8")
                
                input_data = json.loads(video_bytes)
                
                if "video_path" in input_data:
                    video_path = Path(input_data["video_path"])
                elif "video_base64" in input_data:
                    import base64
                    video_path = self._save_video(input_data["video_base64"])
                else:
                    raise ValueError("Input must contain 'video_path' or 'video_base64'")

          
                config = ASRConfig(
                    chunk_size=input_data.get("chunk_size", self.chunk_size),
                    left_context_size=input_data.get("left_context_size", self.left_context),
                    right_context_size=input_data.get("right_context_size", self.right_context),
                    total_batch_duration=input_data.get("total_batch_duration", 800),
                    sample_rate=input_data.get("sample_rate", 16000),
                    num_extraction_workers=input_data.get("num_extraction_workers", 4),
                    num_asr_workers=input_data.get("num_asr_workers", 4)
                )

        
                audio_path = self.temp_dir / f"{uuid.uuid4().hex}.wav"
                success = self.processor.extract_audio(video_path, audio_path, config.sample_rate)
                
                if not success:
                    raise RuntimeError("Audio extraction failed")

                
                result = self.processor.process_audio(audio_path, str(video_path), config)

            
                output_dict = {
                    "tokens": [
                        {
                            "text": t.text,
                            "start": t.start,
                            "end": t.end,
                            "start_frame": t.start_frame,
                            "end_frame": t.end_frame
                        }
                        for t in result.tokens
                    ],
                    "processing_time": result.processing_time_seconds,
                    "audio_duration": result.audio_duration_seconds
                }

            
                if audio_path.exists():
                    audio_path.unlink()
                if "video_base64" in input_data and video_path.exists():
                    video_path.unlink()

                output_tensor = pb_utils.Tensor(
                    "transcription",
                    np.array([json.dumps(output_dict)], dtype=object)
                )
                
                responses.append(pb_utils.InferenceResponse([output_tensor]))

            except Exception as e:
                error_message = str(e)
                responses.append(pb_utils.InferenceResponse(
                    [pb_utils.Tensor("transcription", np.array([json.dumps({"error": error_message})], dtype=object))],
                    error=error_message
                ))

        return responses

    def _save_video(self, base64_data: str) -> Path:
        """Save base64 video to temp file."""
        video_path = self.temp_dir / f"{uuid.uuid4().hex}.mp4"
        with open(video_path, "wb") as f:
            f.write(base64.b64decode(base64_data))
        return video_path

    def finalize(self):
        """Cleanup when model is unloaded."""
        if hasattr(self, 'processor'):
            import torch
            if self.device == "cuda":
                torch.cuda.empty_cache()