import torch
from contextlib import contextmanager
from typing import Iterator
import ffmpeg
import numpy as np
from urllib.parse import urlparse



def split_minio_url(uri:str):
    parsed = urlparse(uri)
    bucket = parsed.netloc
    object_name = parsed.path.lstrip("/")
    return bucket, object_name


def get_frames_fast(video_file_path: str, width=48, height=27) -> np.ndarray:
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


def preprocess_input_client(batch: np.ndarray):
    batch = np.transpose(batch, (3, 0, 1, 2))
    batch = np.expand_dims(batch, axis=0)
    return batch

def postprocess_output_client(one_hot: np.ndarray) -> np.ndarray:
    tensor = torch.tensor(one_hot)

    if isinstance(tensor, tuple):
        tensor = tensor[0]

    prediction = torch.sigmoid(tensor[0]).cpu().numpy()
    return prediction[25:75]


