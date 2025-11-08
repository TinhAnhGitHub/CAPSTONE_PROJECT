"""
This file holds basic utilities to help agents recognize and convert between
video frames, timestamps, and durations.
"""
import asyncio
import base64
import mimetypes
from typing import Tuple, cast
from agentic_ai.tools.schema.artifact import VideoInterface, ImageObjectInterface, SegmentObjectInterface
from typing import Annotated
from agentic_ai.tools.clients.postgre.client import PostgresClient
from ingestion.core.artifact.schema import ASRArtifact
from agentic_ai.tools.clients.minio.client import StorageClient
from collections import defaultdict
from ingestion.prefect_agent.service_asr.core.schema import  ASRResult
import re
from .helper import extract_s3_minio_url, time_range_overlap, time_to_seconds

from .registry import tool_registry


@tool_registry.register(
    category="Utility/Time",
    tags=["timecode", "frame", "conversion"],
    dependencies=[]
)
def frame_to_timecode(frame_index: int, fps: float) -> str:
    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")
    total_seconds = frame_index / fps

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"



@tool_registry.register(
    category="Utility/Time",
    tags=["timecode", "frame", "conversion"],
    dependencies=[]
)
def timecode_to_frame(timecode: str, fps: float) -> int:
    """
    Convert a timecode string (HH:MM:SS.sss) into a frame index given a frame rate (FPS).

    Args:
        timecode (str): Timecode string, e.g., "00:01:23.456" or "01:10:05.003".
        fps (float): Frames per second. Must be > 0.

    Returns:
        int: The corresponding frame index (rounded to nearest integer).

    Raises:
        ValueError: If the timecode format is invalid or FPS <= 0.
    """
    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")
    match = re.match(r"^(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)$", timecode)
    if not match:
        raise ValueError(f"Invalid timecode format: '{timecode}'. Expected 'HH:MM:SS.sss'.")
    hours, minutes, seconds = match.groups()
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)    
    frame_index = round(total_seconds * fps)
    return frame_index



@tool_registry.register(
    category="Utility/Time",
    tags=["time", "frame", "conversion", "video"],
    dependencies=[]
)
def from_index_to_time(
    video: VideoInterface,
    frame_index: int
) -> str:
    """
    Return the exact timestamp (in ISO 8601 format) for a given frame index.
    Args:
        video (VideoObject): The reference video object.
        frame_index (int): The frame index to convert.
    Returns:
        str: ISO-format timestamp corresponding to the frame position.
    """
    return frame_to_timecode(frame_index=frame_index, fps=video.fps)


@tool_registry.register(
    category="Utility/Time",
    tags=["time", "frame", "conversion", "video"],
    dependencies=[]
)
def from_range_index_to_range_time(
    video: VideoInterface,
    start_frame_index: int,
    end_frame_index: int
) -> str:
    """
    Convert a frame range into a corresponding time range.
    Args:
        video (VideoInterface): The reference video object.
        start_frame_index (int): The first frame index in the range.
        end_frame_index (int): The last frame index in the range.
    Returns:
        Tuple[str, str]: Start and end timestamps (ISO 8601 format).
    """
    start,end = frame_to_timecode(frame_index=start_frame_index, fps=video.fps),frame_to_timecode(frame_index=end_frame_index, fps=video.fps)
    return f"Result: ({start},{end})"


@tool_registry.register(
    category="Utility/Time",
    tags=["time", "frame", "conversion", "video"],
    dependencies=[]
)
def from_time_to_index(
    video: VideoInterface,
    time: str
) -> int:
    """
    Return the frame index corresponding to a given timestamp.
    Args:
        video (VideoInterface): The reference video object.
        time (str): Timestamp in ISO 8601 format (e.g., '2025-10-24T12:34:56.789Z').
    Returns:
        int: Frame index closest to the given time.
    """
    return timecode_to_frame(timecode=time, fps=video.fps)
    


@tool_registry.register(
    category="Utility/Time",
    tags=["time", "frame", "conversion", "video"],
    dependencies=[]
)
def from_range_time_to_range_index(
    video: VideoInterface,
    start_time: str,
    end_time: str
) -> str:
    """
    Convert a time range into a corresponding frame index range.
    Args:
        video (VideoInterface): The reference video object.
        start_time (str): Start timestamp in ISO 8601 format.
        end_time (str): End timestamp in ISO 8601 format.
    Returns:
        Tuple[int, int]: Start and end frame indices.
    """
    start,end = timecode_to_frame(timecode=start_time, fps=video.fps),timecode_to_frame(timecode=end_time, fps=video.fps)
    return f"Result: ({start},{end})"


@tool_registry.register(
    category="Utility/IO",
    tags=["minio", "image", "read", "binary"],
    dependencies=["minio_client"]
)
def read_image(image_interface: ImageObjectInterface, minio_client: StorageClient) -> tuple[str, str]:
    """
    Read image, return base64 and mime/type
    """
    minio_path = image_interface.minio_path
    bucket, object_name = extract_s3_minio_url(minio_path)
    image_bytes = cast(bytes, minio_client.get_object(bucket=bucket, object_name=object_name))
    mime_type, _ = mimetypes.guess_type(object_name)
    mime_type = mime_type or "application/octet-stream"
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return base64_str, mime_type

@tool_registry.register(
    category="Utility/IO",
    tags=["minio", "postgres", "segment", "read", "video"],
    dependencies=["minio_client", "postgres_client"]
)
async def read_segment(
    segment_interface: SegmentObjectInterface, 
    minio_client: StorageClient, 
    postgres_client: PostgresClient
) -> Tuple[bytes, str]:
    """
    Fetch a video segment and return its extension and bytes.
    """

    video_artifact = await postgres_client.get_artifact(artifact_id=segment_interface.related_video_id)
    if video_artifact is None:
        raise ValueError(f"Video artifact with id {segment_interface.related_video_id} should exist but does not.")

    extension = (video_artifact.artifact_metadata or {}).get("extension")
    if not extension:
        raise KeyError(f"Missing 'extension' in artifact_metadata for {video_artifact.artifact_id}")

    video_url = video_artifact.minio_url
    bucket_video, object_video_name = extract_s3_minio_url(video_url)

    loop = asyncio.get_event_loop()
    video_bytes = await loop.run_in_executor(
        None,
        lambda: minio_client.get_object(bucket=bucket_video, object_name=object_video_name)
    )
    video_bytes = cast(bytes, video_bytes)
    mime_type = mimetypes.types_map.get(f".{extension}", "application/octet-stream")
    return video_bytes, mime_type
     
    

@tool_registry.register(
    category="Utility/ASR",
    tags=["asr", "context", "multimodal", "postgres", "minio"],
    dependencies=["postgres_client", "minio_client"]
)
async def get_related_asr_from_image(
    image_object_interface: Annotated[ImageObjectInterface, "Image or segment to contextualize with ASR."],
    window_seconds: Annotated[
        float,
        "Time window around artifact (± seconds for transcript snippet)."
        "If 10, then the return "
    ],
    postgres_client: Annotated[PostgresClient, "Auto-provided."],
    minio_client: Annotated[StorageClient, "Auto-provided."],
)->str:
    parent_video_id = image_object_interface.related_video_id

    video_asr_artifacts = await postgres_client.get_children_artifact(
        artifact_id=parent_video_id,
        filter_artifact_type=[ASRArtifact.__name__]
    )  
    
    video_asr_artifact = video_asr_artifacts[0]
    bucket_name, object_name = extract_s3_minio_url(video_asr_artifact.minio_url)
    minio_object = cast(dict, minio_client.read_json(bucket=bucket_name, object_name=object_name))
    asr_object = ASRResult.model_validate(minio_object)

    asr_tokens = asr_object.tokens
    print(asr_tokens)

    timestamp = image_object_interface.timestamp

    ts_center = time_to_seconds(timestamp)
    window_start = ts_center - window_seconds
    window_end = ts_center + window_seconds

    print("ASR range:", asr_tokens[0].start, "→", asr_tokens[-1].end)
    print("Image timestamp:", timestamp)
    
    snippet_tokens = [
        t for t in asr_tokens
        
        if time_range_overlap(
            start_input=window_start,
            end_input=window_end,
            start_range=time_to_seconds(t.start),
            end_range=time_to_seconds(t.end),
            iou=0.2
        )
    ]

    snippet = """
    Here are the asr captured around the image at timestamp/frame_index: {image_timestamp}/{image_frame_index} around the window seconds: {window_seconds}
    {context}
    The asr might have irrelevant events/context, so just focus on the related asr segments!
    The context around:{context}
    """

    context = []
    for token in snippet_tokens:
        context_token = f"""
        ---------------------
        Start time/index: {token.start}/{token.start_frame} - End time/index: {token.end}/{token.end_frame}
        ASR content: {token.text}
        ---------------------
        """
        context.append(context_token)
    
    return_snippet = snippet.format(
        image_timestamp=image_object_interface.timestamp,
        image_frame_index=image_object_interface.frame_index,
        window_seconds=window_seconds,
        context='\n\n'.join(context)
    )

    return return_snippet



@tool_registry.register(
    category="Utility/ASR",
    tags=["asr", "context", "multimodal", "postgres", "minio"],
    dependencies=["postgres_client", "minio_client"]
)
async def get_related_asr_from_segment(
    segment_interface: SegmentObjectInterface,
    window_seconds: Annotated[
        float,
        "Time window around artifact (± seconds for transcript snippet)."
        "If 10, then the return "
    ],
    postgres_client: Annotated[PostgresClient, "Auto-provided."],
    minio_client: Annotated[StorageClient, "Auto-provided."],
)-> str:
    parent_video_id = segment_interface.related_video_id
    video_asr_artifacts = await postgres_client.get_children_artifact(
        artifact_id=parent_video_id,
        filter_artifact_type=[ASRArtifact.__name__]
    ) 

    video_asr_artifact = video_asr_artifacts[0]
    bucket_name, object_name = extract_s3_minio_url(video_asr_artifact.minio_url)
    minio_object = cast(dict, minio_client.read_json(bucket=bucket_name, object_name=object_name))
    asr_object = ASRResult.model_validate(minio_object)
    asr_tokens = asr_object.tokens

    segment_start_time = time_to_seconds(segment_interface.start_time)
    segment_end_time = time_to_seconds(segment_interface.end_time)
    window_start = segment_start_time - window_seconds
    window_end = segment_end_time + window_seconds

    snippet_tokens = [
        t for t in asr_tokens
        if time_range_overlap(
            start_input=window_start,
            end_input=window_end,
            start_range=time_to_seconds(t.start),
            end_range=time_to_seconds(t.end),
            iou=0.2
        )
    ]

    snippet = """
    ASR transcript context around the segment:

    ▶ Segment range: {segment_start_time} → {segment_end_time}
    ▶ Frame range: {segment_start_frame} → {segment_end_frame}
    ▶ Context window: ±{window_seconds} seconds

    --------------------- TRANSCRIPT CONTEXT ---------------------
    {context}
    --------------------------------------------------------------

    Note: Some ASR lines may include adjacent context beyond the target segment.
    Focus on lines semantically aligned with the segment’s content.
    """

    context = []
    for token in snippet_tokens:
        context_token = f"""
        Start time/index: {token.start}/{token.start_frame}
        End time/index:   {token.end}/{token.end_frame}
        ASR content:      {token.text}
        """
        context.append(context_token.strip())

    return_snippet = snippet.format(
        segment_start_time=segment_interface.start_time,
        segment_end_time=segment_interface.end_time,
        segment_start_frame=segment_interface.start_frame_index,
        segment_end_frame=segment_interface.end_frame_index,
        window_seconds=window_seconds,
        context="\n\n".join(context)
    )

    return return_snippet

