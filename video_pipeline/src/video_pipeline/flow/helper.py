import tempfile
import ffmpeg


def extract_single_audio_segment(
    video_path: str,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> str:
    start_sec = start_frame / fps
    duration_sec = (end_frame - start_frame) / fps

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    (
        ffmpeg
        .input(video_path, ss=start_sec, t=max(duration_sec, 0.01))
        .output(
            tmp.name,
            acodec="pcm_s16le",
            ac=1,
            ar="16000",
            vn=None,
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return tmp.name
