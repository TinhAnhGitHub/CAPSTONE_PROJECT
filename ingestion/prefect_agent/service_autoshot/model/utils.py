import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional

import ffmpeg
import numpy as np
import requests
from urllib.parse import urlparse



def get_frames(video_file_path: str, width: int = 48, height: int = 27) -> np.ndarray:
    """
    Extract frames from video 
    Args:
        video_file_path (str): Path to the video file.
        width (int): Width of the extracted frame. Default is 48
        height (int): Height of the extracted frames. Default is 27
    Returns:
        np.ndarray: Array of video frames
    """
    try:
        out, _ = (
             ffmpeg
            .input(video_file_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}')
            .run(capture_stdout=True, capture_stderr=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        return video
    except ffmpeg.Error as e:
        print(f"ffmpeg error: {e.stderr.decode()}")
        raise
    except Exception as e:
        print(f"Error in get_frames: {str(e)}")
        raise




def get_batches(
    frames: np.ndarray
) -> Iterator[np.ndarray]:
    
    if len(frames) == 0:
        return

    remainder = 50 - (len(frames) % 50)
    if remainder == 50:
        remainder = 0
    
    pad_start = 25
    pad_end = remainder + 25

    padded_frames = np.concatenate(
        [
            np.repeat(frames[:1], pad_start, axis=0),
            frames,
            np.repeat(frames[-1:], pad_end, axis=0)
        ], axis=0   
    )

    batchsize = 100
    stride = 50 
    for i in range(
        0, len(padded_frames) - stride, stride
    ):
        batch = padded_frames[i:i + batchsize]
        if len(batch) < batchsize:
            padded = batchsize - len(batch)
            batch = np.concatenate(
                [
                    batch,
                    np.repeat(batch[-1:], repeats=padded, axis=0    )
                ], axis=0   
            )
        yield batch.transpose(
            (
                1, 2, 3, 0 #* height, width, color, frames
            )
        )



RESET = "\x1b[0m"
COLORS = {
    logging.DEBUG: "\x1b[38;20m",     
    logging.INFO: "\x1b[32;20m",      
    logging.WARNING: "\x1b[33;20m",   
    logging.ERROR: "\x1b[31;20m",     
    logging.CRITICAL: "\x1b[31;1m",   
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        log_color = COLORS.get(record.levelno, RESET)
        
        filepath = Path(record.pathname).name
        lineno = record.lineno
        
        log_fmt = (
            f"{log_color}[%(levelname)s]{RESET} "
            f"%(asctime)s "
            f"{log_color}%(name)s{RESET} "
            f"{filepath}:{lineno} - "
            f"%(message)s"
        )
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def setup_logger(level=logging.INFO):
    """Configure root logger with colors and file/line info."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter())
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)





def resolve_video_url(video_url: str) -> str:
    
    parsed = urlparse(video_url)
    if parsed.scheme in {"http", "https"}:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        
        with requests.get(video_url, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                tmp.write(chunk)
        tmp.close()
        return tmp.name
    return video_url
