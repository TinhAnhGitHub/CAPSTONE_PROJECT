"""
This file holds basic utilities to help agents recognize and convert between
video frames, timestamps, and durations.
"""
import asyncio
from datetime import datetime
import base64
import mimetypes
from typing import Tuple, cast
from agentic_ai.tools.schema.artifact import VideoObject, ImageObjectInterface, SegmentObjectInterface
from typing import Annotated
from agentic_ai.tools.clients.postgre.client import PostgresClient
from ingestion.core.artifact.schema import ASRArtifact
from agentic_ai.tools.clients.minio.client import StorageClient
from collections import defaultdict
from ingestion.prefect_agent.service_asr.core.schema import  ASRInferenceResponse
import re
from .helper import extract_s3_minio_url

from .registry import tool_registry



@tool_registry.register(
    category='Utility',
    tags=["enhance", "visual"],
    dependencies=None
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
    category='Utility',
    tags=["enhance", "visual"],
    dependencies=None
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
    category='Utility',
    tags=["enhance", "visual"],
    dependencies=None
)
def organize_images(
        list_images: Annotated[list[ImageObjectInterface], "List of ImageObjectInterface"], 
        query:Annotated[str, "The query that you just search for"]) -> dict:
    """
    This function will return readable representation of ImageObjectInterface, which help you to better understand the results.
    """
    result_dict: dict[str, list[dict]] = defaultdict(list)
    
    for image in list_images:
        result_dict[image.related_video_id].append({
            "frame_index": image.frame_index,
            "timestamp": image.timestamp,
            "caption": image.caption_info,
            "score": round(image.score, 4) if image.score else "No score",
            "minio_path": image.minio_path,
            "query_relation": f"Match for query: '{query}'"
        })
    
    readable_result = {
        "type": "visual_search_result",
        "query": query,
        "summary": f"Retrieved {len(list_images)} visually similar frames across {len(result_dict)} videos.",
        "results": result_dict
    }
    
    return readable_result


@tool_registry.register(
    category='Utility',
    tags=["enhance", "visual"],
    dependencies=None
)
def organize_segments(
    list_segments: Annotated[list["SegmentObjectInterface"], "List of SegmentObjectInterface"], 
    query: Annotated[str, "The query that you just searched for"]
) -> dict:
    """
    Organize retrieved SegmentObjectInterface objects into a structured,
    descriptive, and interpretable representation grouped by their related video ID.

    This representation helps downstream agents or evaluators understand
    what parts of each video are most semantically relevant to the query.
    """

    result_dict: dict[str, list[dict]] = defaultdict(list)

    for segment in list_segments:
        result_dict[segment.related_video_id].append({
            "start_frame_index": segment.start_frame_index,
            "end_frame_index": segment.end_frame_index,
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "caption": segment.caption_info,
            "score": round(segment.score, 4) if segment.score else "No score",
            "duration": f"{segment.start_time} → {segment.end_time}",
            "query_relation": f"Segment semantically related to query: '{query}'"
        })

    for video_id in result_dict:
        result_dict[video_id].sort(key=lambda x: x["score"], reverse=True)

    readable_result = {
        "type": "visual_segment_search_result",
        "query": query,
        "summary": (
            f"Retrieved {len(list_segments)} relevant segments across "
            f"{len(result_dict)} videos based on semantic similarity to the query."
        ),
        "results": result_dict
    }

    return readable_result


@tool_registry.register(
    category='Utility',
    tags=["enhance", "visual"],
    dependencies=None
)
def from_index_to_time(
    video: VideoObject,
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
    category='Utility',
    tags=["enhance", "visual"],
    dependencies=None
)
def from_range_index_to_range_time(
    video: VideoObject,
    start_frame_index: int,
    end_frame_index: int
) -> Tuple[str, str]:
    """
    Convert a frame range into a corresponding time range.
    Args:
        video (VideoObject): The reference video object.
        start_frame_index (int): The first frame index in the range.
        end_frame_index (int): The last frame index in the range.
    Returns:
        Tuple[str, str]: Start and end timestamps (ISO 8601 format).
    """
    return frame_to_timecode(frame_index=start_frame_index, fps=video.fps),frame_to_timecode(frame_index=end_frame_index, fps=video.fps)


@tool_registry.register(
    category='Utility',
    tags=["enhance", "visual"],
    dependencies=None
)
def from_time_to_index(
    video: VideoObject,
    time: str
) -> int:
    """
    Return the frame index corresponding to a given timestamp.
    Args:
        video (VideoObject): The reference video object.
        time (str): Timestamp in ISO 8601 format (e.g., '2025-10-24T12:34:56.789Z').
    Returns:
        int: Frame index closest to the given time.
    """
    return timecode_to_frame(timecode=time, fps=video.fps)
    


@tool_registry.register(
    category='Utility',
    tags=["enhance", "visual"],
    dependencies=None
) 
def from_range_time_to_range_index(
    video: VideoObject,
    start_time: str,
    end_time: str
) -> Tuple[int, int]:
    """
    Convert a time range into a corresponding frame index range.
    Args:
        video (VideoObject): The reference video object.
        start_time (str): Start timestamp in ISO 8601 format.
        end_time (str): End timestamp in ISO 8601 format.
    Returns:
        Tuple[int, int]: Start and end frame indices.
    """
    return timecode_to_frame(timecode=start_time, fps=video.fps), timecode_to_frame(timecode=end_time, fps=video.fps)


@tool_registry.register(
    category='Utility',
    tags=["enhance", "visual"],
    dependencies=['minio_client']
) 
def read_image(image_interface: ImageObjectInterface, minio_client: StorageClient) -> tuple[str, str]:
    """
    Read image, return base64 and mime/type
    """
    minio = image_interface.minio_path
    bucket, object_name = extract_s3_minio_url(minio)
    image_bytes = cast(bytes, minio_client.get_object(bucket=bucket, object_name=object_name))
    mime_type, _ = mimetypes.guess_type(object_name)
    mime_type = mime_type or "application/octet-stream"
    base64_str = base64.b64encode(image_bytes).decode("utf-8")
    return base64_str, mime_type

@tool_registry.register(
    category='Utility',
    tags=["enhance", "visual"],
    dependencies=['minio_client', 'postgres_client']
) 
async def read_segment(
    segment_interface: SegmentObjectInterface, 
    minio_client: StorageClient, 
    postgres_client: PostgresClient
) -> Tuple[str, bytes]:
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
    return extension, video_bytes
     
    

@tool_registry.register(
    category='Utility',
    tags=["enhance", "visual"],
    dependencies=['minio_client', 'postgres_client']
) 
async def get_related_asr(
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
    # should return 1
    video_asr_artifact = video_asr_artifacts[0]
    bucket_name, object_name = extract_s3_minio_url(video_asr_artifact.minio_url)
    minio_object = cast(dict, minio_client.read_json(bucket=bucket_name, object_name=object_name))
    asr_object = ASRInferenceResponse.model_validate(minio_object)

    asr_tokens = asr_object.result.tokens

    timestamp = image_object_interface.timestamp

    def time_to_seconds(time_str: str) -> float:
        t = datetime.strptime(time_str, "%H:%M:%S.%f")
        return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

    def time_range_overlap(
        start_input: str,
        end_input:str,
        start_range: str,
        end_range:str,
        iou=0.1,
    ):
        """
        Return True if the iou between the range of input and "range" exceed 0.2, or the input is inside the range
        """
        s1, e1 = time_to_seconds(start_input), time_to_seconds(end_input)
        s2, e2 = time_to_seconds(start_range), time_to_seconds(end_range)
        inter = max(0, min(e1, e2) - max(s1, s2))
        union = max(e1, e2) - min(s1, s2)
        return (inter / union) >= iou or (s1 >= s2 and e1 <= e2)

    ts_center = time_to_seconds(timestamp)
    window_start = ts_center - window_seconds
    window_end = ts_center + window_seconds
    
    snippet_tokens = [
        t for t in asr_tokens
        if time_range_overlap(
            start_input=f"{timestamp}",
            end_input=f"{timestamp}",
            start_range=t.start,
            end_range=t.end,
            iou=0.0
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

    
    
    






# async def link_artifacts_by_lineage(
#     start_artifact_id: Annotated[str, "Starting artifact ID (e.g., image artifact_id)."],
#     depth: Annotated[int, "Lineage traversal depth (default: 2).", default=2],
#     filter_types: Annotated[list[str], "Artifact types to include (e.g., ['ImageArtifact', 'ASRArtifact']).", default=[]],
#     postgre_client: Annotated[PostgresClient, "Auto-provided."],
# ) -> list[ArtifactMetadata]:
    




async def find_similar_moments_from_image(
    reference: ImageObjectInterface | SegmentObjectInterface,
    similarity_threshold: float = 0.8,
    search_scope: Annotated[Literal["same_video", "all_videos"], "Where to search"],
    **clients
) -> list[tuple[float, ImageObjectInterface | SegmentObjectInterface]]:
    """
    Find moments visually/semantically similar to a reference.
    Useful for finding repeated actions, similar scenes, or callbacks.
    """
    






# async def explore_window_events(
#     focal_artifact: Annotated[
#         ImageObjectInterface | SegmentObjectInterface,
#         "Central image/segment to expand from."
#     ],
#     window_seconds: Annotated[float, "± Time window (seconds) for context.", default=30.0],
#     include_segments: Annotated[bool, "Fetch segments too (for hybrid view).", default=True],
#     llm_prompt_template: Annotated[str, "Custom prompt for LLM event generation.", default="Summarize events around {focal}: {artifacts_list}"],
#     postgre_client: Annotated[PostgresClient, "Auto-provided."],
#     minio_client: Annotated[StorageClient, "Auto-provided."],
#     # Note: For real LLM, inject ExternalEncodeClient or call Grok API; here, placeholder concat
# ) -> Dict[str, Any]:
#     """
#     Simple agent tool: Fetch window → LLM for events.
    
#     Chains fetch → list format → LLM summary (placeholder: simple join; replace with ExternalEncodeClient.encode_text for real LLM).
#     """





# async def rerank_window_artifacts_by(
#     artifacts: Annotated[List[Union[ImageObjectInterface, SegmentObjectInterface]], "Artifacts from window fetch."],
#     relevance_query: Annotated[str, "Query to rerank by (e.g., 'action scene')."],
#     top_k: Annotated[int, "Keep top N after rerank.", default=5],
#     external_client: Annotated[ExternalEncodeClient, "For query embedding."],
# ) -> List[Union[ImageObjectInterface, SegmentObjectInterface]]:
#     """
#     Rerank artifacts by cosine sim to query embedding (textual/caption focus). Rerank by LLM
#     """
   