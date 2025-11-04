import cv2
from pathlib import Path
from fastapi import UploadFile
from datetime import timedelta
from typing import BinaryIO
from fractions import Fraction
import ffmpeg 
import hashlib
def valid_video_files(file: UploadFile) -> bool:
    video_mime_types = {
        "video/mp4",
        "video/mpeg",
        "video/avi",
        "video/quicktime",   
        "video/x-msvideo",
        "video/x-ms-wmv",
        "video/x-flv",
        "video/webm",
        "video/3gpp",
        "video/3gpp2",
        "application/octet-stream",  
    }
    video_extensions = {
        ".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv",
        ".webm", ".3gp", ".mpeg", ".mpg"
    }

    content_type = (file.content_type or "").lower()
    filename = file.filename or ""

    if content_type not in video_mime_types and not any(filename.endswith(ext) for ext in video_extensions):
       return False
    return True


def get_video_metadata(video_filename: str, file: BinaryIO) -> dict:
    extension = Path(video_filename).suffix
    file.seek(0)
    import tempfile
    with tempfile.NamedTemporaryFile(delete=True, suffix=extension) as tmp:
        tmp.write(file.read())
        tmp.flush()

        probe = ffmpeg.probe(tmp.name)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None
        )
        if video_stream is None:
            return {}
        
        tmp.seek(0)
        md5 = hashlib.md5(tmp.read()).hexdigest()
        fps = float(Fraction(video_stream["avg_frame_rate"])) if "avg_frame_rate" in video_stream else None

        file.seek(0)
        return {
            "filename": video_filename,
            "codec": video_stream.get("codec_name"),
            "fps": fps,
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "size_bytes": int(probe["format"]["size"]),
            "checksum_md5": md5,
            'extension': extension
        }
    


def get_video_fps(video_path: str) -> float:
    """Return the FPS (frames per second) of a video file using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    return fps 


import cv2
from datetime import timedelta

import cv2
from datetime import timedelta

def get_video_duration_cv2(path: str) -> str:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0:
        raise ValueError("Invalid FPS retrieved; possibly corrupted file.")

    duration_seconds = frame_count / fps
    td = timedelta(seconds=duration_seconds)

    total_seconds = td.total_seconds()
    hours, remainder = divmod(int(total_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03d}"

