import math
from typing import Iterator
import numpy as np
from urllib.parse import urlparse



def split_minio_url(uri:str):
    parsed = urlparse(uri)
    bucket = parsed.netloc
    object_name = parsed.path.lstrip("/")
    return bucket, object_name


def get_frames_fast(video_file_path: str, width=48, height=27) -> np.ndarray:
    import ffmpeg

    stream = (
        ffmpeg
        .input(video_file_path, threads=0)
        .filter('scale', width, height, flags='fast_bilinear')
        .output(
            'pipe:',
            format='rawvideo',
            pix_fmt='rgb24',
            vsync='vfr'
        )
    )

    out, _ = stream.run(capture_stdout=True, quiet=True)
    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    return video

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
                1, 2, 3, 0 
            )
        )


def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    predictions = (predictions > threshold).astype(np.uint8)
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i]) #type:ignore

    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)
    return np.array(scenes, dtype=np.int32)


def compute_outlier_threshold(lengths: list[int]) -> float:
    arr = np.array(lengths)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    return float(q3 + 1.5 * iqr)


def compute_target_size(lengths: list[int], threshold: float) -> int:
    arr = np.array(lengths)
    normal = arr[arr <= threshold]
    return max(1, int(np.percentile(normal, 75)))


def split_long_segment(start: int, end: int, target_size: int) -> list[tuple[int, int]]:
    length = end - start
    if length <= target_size:
        return [(start, end)]

    num_chunks = math.ceil(length / target_size)
    chunk_size = length / num_chunks
    return [
        (round(start + i * chunk_size), round(start + (i + 1) * chunk_size))
        for i in range(num_chunks)
    ]


def split_outlier_scenes(scenes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Split unusually long scenes into near-even chunks.

    Uses the same IQR-based outlier rule as ``local/test_new_algo.ipynb``:
    scenes above ``Q3 + 1.5 * IQR`` are split using a target size based on the
    75th percentile of the non-outlier scene lengths.
    """
    if not scenes:
        return []

    lengths = [end - start for start, end in scenes]
    threshold = compute_outlier_threshold(lengths)
    target_size = compute_target_size(lengths, threshold)

    final_scenes: list[tuple[int, int]] = []
    for start, end in scenes:
        if end - start > threshold:
            final_scenes.extend(split_long_segment(start, end, target_size))
        else:
            final_scenes.append((start, end))
    return final_scenes


def enforce_min_scene_segments(
    scenes: list[tuple[int, int]],
    min_segments: int,
) -> list[tuple[int, int]]:
    """Evenly subdivide the current scene timeline when too few scenes exist."""
    if min_segments <= 0 or not scenes or len(scenes) >= min_segments:
        return scenes

    ordered_scenes = sorted(scenes)
    start_frame = ordered_scenes[0][0]
    end_frame = ordered_scenes[-1][1]
    total_frames = max(1, end_frame - start_frame)

    boundaries = [
        start_frame + round(index * total_frames / min_segments)
        for index in range(min_segments + 1)
    ]
    boundaries[0] = start_frame
    boundaries[-1] = end_frame

    return [
        (boundaries[index], boundaries[index + 1])
        for index in range(min_segments)
    ]


def preprocess_input_client(batch: np.ndarray):
    batch = np.transpose(batch, (3, 0, 1, 2))
    batch = np.expand_dims(batch, axis=0)
    return batch

def postprocess_output_client(one_hot: np.ndarray) -> np.ndarray:
    if isinstance(one_hot, tuple):
        one_hot = one_hot[0]

    prediction = 1 / (1 + np.exp(-one_hot[0]))
    return prediction[25:75]
