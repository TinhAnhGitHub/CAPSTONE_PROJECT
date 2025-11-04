import os
import torch
import yaml
import cv2
import time
import uvicorn
from pathlib import Path
from typing import Dict, List, Optional
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from moviepy import VideoFileClip
from chunkformer import ChunkFormerModel
from service_asr.core.schema import ASRConfig, ASRResult, TimestampedToken


class ASRProcessor:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device:str = device
        self.model: ChunkFormerModel = None #type:ignore
        self.char_dict: Optional[Dict[int, str]] = None
        self.model_name = model_name
        self._load_model()

    def _load_model(self) -> None:
        self.model = ChunkFormerModel.from_pretrained(self.model_name).to(self.device) #type:ignore
    

    @staticmethod
    def extract_fps(video_path: str) -> float:
        """Extract FPS from video file."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    @staticmethod
    def time_to_frames(time_str: str, fps: float) -> int:
        """Convert HH:MM:SS:ms to frame number."""
        time_str = time_str.strip().replace('.', ':')
        parts = time_str.split(':')
        if len(parts) != 4:
            raise ValueError(f"Invalid time format: {time_str} (expected HH:MM:SS:ms)")
        h, m, s, ms = map(int, parts)
        total_seconds = h * 3600 + m * 60 + s + ms / 1000.0
        return int(round(total_seconds * fps))

    def time_transform(self, tokens: List[Dict[str, str]], video_path: str) -> List[TimestampedToken]:
        """Convert timestamps to frames and create TimestampedToken objects."""
        fps = self.extract_fps(video_path)
        result = []
        
        for token in tokens:
            start_frame = self.time_to_frames(token['start'], fps)
            end_frame = self.time_to_frames(token['end'], fps)
            
            result.append(TimestampedToken(
                text=token['decode'],
                start=token['start'][::-1].replace(":", ".", 1)[::-1] ,
                end=token['end'][::-1].replace(":", ".", 1)[::-1] ,
                start_frame=f"{start_frame:08d}",
                end_frame=f"{end_frame:08d}"
            ))
        
        return result


    def load_audio(self, audio_path: Path) -> torch.Tensor:
        """Load and preprocess audio file without pydub.

        - Loads with torchaudio
        - Converts to mono, 16 kHz
        - Scales to int16 range to match prior behavior
        """
        waveform, sr = torchaudio.load(str(audio_path))
        # ensure 2D [channels, time]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        # convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # resample if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        # match previous amplitude scale (int16 to float32)
        waveform = (waveform * 32768.0).to(torch.float32)
        return waveform

    def extract_audio(self, video_path: Path, output_path: Path, sample_rate: int = 16000) -> bool:
        """Extract audio from video file."""
        try:
            video = VideoFileClip(str(video_path))
            audio = video.audio
            
            if audio is None:
                return False

            output_path.parent.mkdir(parents=True, exist_ok=True)
            final_wav = output_path.with_suffix('.wav')
            # Force mono (1 channel) and 16-bit PCM at desired sample rate
            audio.write_audiofile(
                str(final_wav),
                fps=sample_rate,          # this sets audio sample rate
                codec='pcm_s16le',        # 16-bit PCM
                ffmpeg_params=['-ac', '1'],  # force mono
                logger=None
            )
            video.close()
            
            return True
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            return False
        

    @torch.no_grad()
    def process_audio(self, audio_path: Path, video_path: str, config: ASRConfig) -> ASRResult:
        print(f"Extracting asr {audio_path}, {video_path}")
        """Process audio file with ASR."""
        start_time = time.time()
        waveform = self.load_audio(audio_path)
        audio_duration = len(waveform[0]) / 16000.0
        transcription: list = self.model.endless_decode( #type:ignore
            audio_path=str(audio_path),
            chunk_size=config.chunk_size,
            left_context_size=config.left_context_size,
            right_context_size=config.right_context_size,
            total_batch_duration=config.total_batch_duration,
            return_timestamps=True
        )
        processing_time = time.time() - start_time
        tokens = self.time_transform(video_path=video_path, tokens=transcription)
        return ASRResult(
            tokens=tokens,
            processing_time_seconds=processing_time,
            audio_duration_seconds=audio_duration
        )
