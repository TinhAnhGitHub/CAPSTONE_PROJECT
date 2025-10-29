import os
import torch
import yaml
import cv2
import time
import uvicorn
from pathlib import Path
from typing import Dict, List, Optional, Any

import torchaudio
import torchaudio.compliance.kaldi as kaldi
from moviepy import VideoFileClip

from service_asr.model.chunkformer.utils.init_model import init_model

from service_asr.model.chunkformer.utils.checkpoint import load_checkpoint
from service_asr.model.chunkformer.utils.file_utils import read_symbol_table
from service_asr.model.chunkformer.utils.ctc_utils import get_output_with_timestamps
from service_asr.model.chunkformer.asr_model import ASRModel
from service_asr.core.schema import ASRConfig, ASRResult, TimestampedToken


class ASRProcessor:
    def __init__(self, model_checkpoint: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.model: ASRModel = None
        self.char_dict: Optional[Dict[int, str]] = None
        self.model_checkpoint = model_checkpoint
        self._load_model()

    def _load_model(self) -> None:
        config_path = os.path.join(self.model_checkpoint, "config.yaml")
        checkpoint_path = os.path.join(self.model_checkpoint, "pytorch_model.bin")
        symbol_table_path = os.path.join(self.model_checkpoint, "vocab.txt")
        

        with open(config_path, 'r') as fin:
            config = yaml.load(fin, Loader=yaml.FullLoader)
        
        self.model = init_model(config, config_path)
        self.model.eval()
        load_checkpoint(self.model, checkpoint_path)
        
        self.model.encoder = self.model.encoder.to(self.device)
        self.model.ctc = self.model.ctc.to(self.device)
        
        symbol_table = read_symbol_table(symbol_table_path)
        self.char_dict = {v: k for k, v in symbol_table.items()}
    

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
                start=token['start'],
                end=token['end'],
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
        
        def get_max_input_context(c: int, r: int, n: int) -> int:
            return r + max(c, r) * (n - 1)

        waveform = self.load_audio(audio_path)
        audio_duration = len(waveform[0]) / 16000.0
        
        offset = torch.zeros(1, dtype=torch.int, device=self.device)
        xs = kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=16000
        ).unsqueeze(0)

        subsampling_factor = self.model.encoder.embed.subsampling_factor
        conv_lorder = self.model.encoder.cnn_module_kernel // 2
        max_length_limited_context = int((config.total_batch_duration // 0.01)) // 2
        multiply_n = max_length_limited_context // config.chunk_size // subsampling_factor
        truncated_context_size = config.chunk_size * multiply_n

        rel_right_context_size = get_max_input_context(
            config.chunk_size, 
            max(config.right_context_size, conv_lorder), 
            self.model.encoder.num_blocks
        ) * subsampling_factor

        hyps = []
        att_cache = torch.zeros((
            self.model.encoder.num_blocks, 
            config.left_context_size, 
            self.model.encoder.attention_heads, 
            self.model.encoder._output_size * 2 // self.model.encoder.attention_heads
        )).to(self.device)
        
        cnn_cache = torch.zeros((
            self.model.encoder.num_blocks, 
            self.model.encoder._output_size, 
            conv_lorder
        )).to(self.device)

        for idx in range(0, xs.shape[1], truncated_context_size * subsampling_factor):
            start = max(truncated_context_size * subsampling_factor * idx, 0)
            end = min(truncated_context_size * subsampling_factor * (idx + 1) + 7, xs.shape[1])

            x = xs[:, start:end + rel_right_context_size]
            x_len = torch.tensor([x[0].shape[0]], dtype=torch.int).to(self.device)

            encoder_outs, encoder_lens, _, att_cache, cnn_cache, offset = \
                self.model.encoder.forward_parallel_chunk(
                    xs=x,
                    xs_origin_lens=x_len,
                    chunk_size=config.chunk_size,
                    left_context_size=config.left_context_size,
                    right_context_size=config.right_context_size,
                    att_cache=att_cache,
                    cnn_cache=cnn_cache,
                    truncated_context_size=truncated_context_size,
                    offset=offset
                )

            encoder_outs = encoder_outs.reshape(1, -1, encoder_outs.shape[-1])[:, :encoder_lens]
            
            if config.chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size < xs.shape[1]:
                encoder_outs = encoder_outs[:, :truncated_context_size]
            
            offset = offset - encoder_lens + encoder_outs.shape[1]
            hyp = self.model.encoder.ctc_forward(encoder_outs).squeeze(0)
            hyps.append(hyp)
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            if config.chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size >= xs.shape[1]:
                break

        hyps = torch.cat(hyps)
        raw_tokens = get_output_with_timestamps([hyps], self.char_dict)[0]
        tokens = self.time_transform(raw_tokens, video_path)
        
        processing_time = time.time() - start_time
        
        return ASRResult(
            tokens=tokens,
            processing_time_seconds=processing_time,
            audio_duration_seconds=audio_duration
        )
