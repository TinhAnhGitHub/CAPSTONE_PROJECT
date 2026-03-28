"""Shared utility functions for video processing tasks."""


def frames_to_timestamp(frame: int, fps: float) -> str:
    """Convert a frame number to a timestamp string HH:MM:SS.mmm."""
    total_seconds = frame / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
