"""
videodeepsearch/tools/implementation/scan/tool.py
This tools allow the agent to act as a human, and a video navigator
"""
from typing import Annotated, Literal, cast
import av
import cv2
import io
from PIL import Image
from videodeepsearch.tools.base.schema import (
    ImageInterface,
    SegmentInterface
)

from videodeepsearch.core.app_state import (
    get_storage_client,
    get_postgres_client
)

from ingestion.core.artifact.schema import SegmentCaptionArtifact, ImageCaptionArtifact

from videodeepsearch.tools.helpers import extract_s3_minio_url, parse_time_safe, timecode_to_frame
from videodeepsearch.tools.base.doc_template.group_doc import GroupName
from videodeepsearch.tools.base.middleware.output import output_image_results, output_segment_results, output_image_bytes
from videodeepsearch.tools.base.doc_template.bundle_template import VIDEO_EVIDENCE_WORKER_BUNDLE
from videodeepsearch.tools.base.types import BundleRoles
from videodeepsearch.tools.base.registry import tool_registry

from videodeepsearch.agent.definition import WORKER_AGENT



HOP_ANNOTATION = Annotated[int, "Define how many steps you want to include/skip. The range is predefined before. For example if you want to see a few segment ahead, it will be like 2 or 3"]

RANGE_ANNOTATION =  Annotated[bool, "If True, then the the hop will including the segments between"]

TIME_DIRECTION = Annotated[
    Literal['forward', 'backward'],
    "Time direction, either you want to hop backward or forward"
]

@tool_registry.register(
    group_doc_name=GroupName.UTILITY,
    bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
    bundle_role_key=BundleRoles.VIDEO_NAVIGATOR,
    output_middleware=output_segment_results,
    input_middleware=None,
    belong_to_agents=[WORKER_AGENT],
)
async def get_segments(
    pivot_segment_related_video_id: str,
    pivot_segment_start_frame: int,
    pivot_segment_end_frame: int,
    hop: HOP_ANNOTATION,
    include_within_range: RANGE_ANNOTATION,
    forward_or_backward: TIME_DIRECTION,
) -> list[SegmentInterface]:
    """
    Navigate to adjacent segments before/after a reference segment.
    
    Retrieves surrounding video segments relative to a known segment, enabling
    temporal exploration of video content. Like "turning pages" in a video book.
    
    **When to use:**
    - Found a promising segment and want to see what happens before/after
    - Need temporal context around a matching segment
    - Verifying if an event continues across multiple segments
    - Exploring scene boundaries and transitions
    
    **Typical workflow:**
    1. Find segment via search tool
    2. View segment details to confirm relevance
    3. Call this tool to see adjacent segments
    4. Inspect adjacent context to verify event boundaries

    """
    minio_client = get_storage_client()
    postgres_client = get_postgres_client()

    parent_video_id = pivot_segment_related_video_id
    children_artifact = await postgres_client.get_children_artifact(artifact_id=parent_video_id, filter_artifact_type=[SegmentCaptionArtifact.__name__])

    segments: list[SegmentInterface] = []
    for child in children_artifact:
        minio_path = child.minio_url
        bucket, object_name = extract_s3_minio_url(minio_path)
        json_dict = minio_client.read_json(bucket=bucket, object_name=object_name)

        if json_dict is None:
            raise ValueError(f"Segment {child.model_dump_json()} can't be found in the Minio storage??")

        caption = json_dict['caption']
        del json_dict['caption']

        segment_artifact = SegmentCaptionArtifact.model_validate(json_dict)
       
        if forward_or_backward == 'forward':
            if  segment_artifact.start_frame >= pivot_segment_end_frame:
                segments.append(
                    SegmentInterface.from_artifact(segment_artifact, caption) 
                )
            segments.sort(key=lambda s: parse_time_safe(s.end_time))
        
        elif forward_or_backward == 'backward':
            if segment_artifact.end_frame <= pivot_segment_start_frame:
                segments.append(
                    SegmentInterface.from_artifact(segment_artifact, caption) 
                )
            segments.sort(key=lambda s: parse_time_safe(s.end_time), reverse=True)
    

    filter_segments = segments[:hop] if include_within_range else [segments[hop-1]]
    return filter_segments




# @tool_registry.register(
#     group_doc_name=GroupName.UTILITY,
#     bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
#     bundle_role_key=BundleRoles.VIDEO_NAVIGATOR,
#     output_middleware=output_image_results,
#     input_middleware=None,
#     belong_to_agents=[WORKER_AGENT]
# )
async def get_image(
    image_related_video_id: str,
    image_frame_index: int,
    hop: HOP_ANNOTATION,
    include_within_range: RANGE_ANNOTATION,
    forward_or_backward: TIME_DIRECTION,
) -> list[ImageInterface]:
    """
    Navigate to adjacent frames before/after a reference image.
    
    Retrieves surrounding frames relative to a known image, enabling frame-by-frame
    exploration. Like stepping through video one frame at a time.
    
    **When to use:**
    - Found a good frame and want to see adjacent frames
    - Checking for motion/action across consecutive frames
    - Verifying object persistence across time
    - Finding best frame in a sequence


    """
    minio_client = get_storage_client()
    postgres_client = get_postgres_client()

    parent_video_id = image_related_video_id
    child_segments = await postgres_client.get_children_artifact(artifact_id=parent_video_id, filter_artifact_type=[ImageCaptionArtifact.__name__])


    filter_segments = []

    for child in child_segments:
        minio_path = child.minio_url
        bucket, object_name = extract_s3_minio_url(minio_path)
        image_id = cast(str,child.lineage_parents[0])
        image_metadata = await postgres_client.get_artifact(artifact_id=image_id)
        if image_metadata is None:
            raise ValueError(f"The image id {image_id} should be exists")

        json_dict = cast(dict,minio_client.read_json(bucket=bucket, object_name=object_name))

        caption_image = json_dict['caption']
        del json_dict['caption']
        image_caption_artifact = ImageCaptionArtifact.model_validate(json_dict)

        if forward_or_backward == 'forward': 
            if image_caption_artifact.frame_index >= image_frame_index:
                filter_segments.append(
                    ImageInterface.from_artifact(image_caption_artifact, caption_image) 
                )
        elif forward_or_backward == 'backward':
            if image_caption_artifact.frame_index <= image_frame_index:
                filter_segments.append(
                    ImageInterface.from_artifact(image_caption_artifact, caption_image) 
                )
        
    filter_segments.sort(key=lambda s: parse_time_safe(s.timestamp), reverse=(forward_or_backward=='backward'))
    return filter_segments[:hop] if include_within_range else [filter_segments[hop-1]]



# @tool_registry.register(
#     group_doc_name=GroupName.UTILITY,
#     bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
#     bundle_role_key=BundleRoles.VIDEO_NAVIGATOR,
#     output_middleware=output_image_bytes,
#     input_middleware=None,
#     belong_to_agents=[WORKER_AGENT]
# )
async def extract_frames_by_time_window(
    video_id: str,
    start_time: Annotated[str, "Start time in HH:MM:SS.sss format"],
    end_time: Annotated[str, "End time in HH:MM:SS.sss format"],
    fps_sample_rate: Annotated[int, "how many frames per second do you want to sample, Ideally 1-2"],
)->list[bytes]:
    """
    Extract raw frames from a video time window at specified frame rate.
    
    Retrieves actual frame images (as bytes) from a video between two timestamps,
    sampled at a custom frame rate. Returns images that can be viewed directly
    by the LLM (via ImageBlock).
    
    **When to use:**
    - Need to visually inspect specific time range in detail
    - Want to show frames directly to LLM for analysis
    - Extracting frames for custom processing
    - Verifying visual content in precise time window
    
    """
    minio_client = get_storage_client()
    postgres_client = get_postgres_client()

    video_artifact = await postgres_client.get_artifact(artifact_id=video_id)
    if video_artifact is None:
        raise ValueError()

    bucket, object_name = extract_s3_minio_url(video_artifact.minio_url)

    video_bytes = minio_client.get_object(bucket=bucket, object_name=object_name)
    if video_bytes is None:
        raise ValueError()
    
    container = av.open(io.BytesIO(video_bytes))
    fps = video_artifact.artifact_metadata['fps']

    start_frame = int(timecode_to_frame(start_time, fps))
    end_frame = int(timecode_to_frame(end_time, fps))


    frames_output: list[bytes] = []
    frame_index = 0 

    stream = container.streams.video[0]
    for frame in container.decode(stream): #type:ignore
        if frame_index < start_frame:
            frame_index += 1
            continue

        if frame_index > end_frame:
            break

        if (frame_index - start_frame) % fps_sample_rate != 0:
            frame_index += 1
            continue

        img = frame.to_ndarray(format="bgr24")
        success, encoded = cv2.imencode(".webp", img)
        if not success:
            frame_index += 1
            continue

        frames_output.append(encoded.tobytes())
        frame_index += 1
    
    container.close()
    return frames_output


# @tool_registry.register(
#     group_doc_name=GroupName.UTILITY,
#     bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
#     bundle_role_key=BundleRoles.VIDEO_NAVIGATOR,
#     output_middleware=output_image_bytes,
#     input_middleware=None,
#     belong_to_agents=[WORKER_AGENT]
# )
async def extract_frame_time(
    video_id: str,
    timestamp: Annotated[str, "Start time in HH:MM:SS.sss format"],
) -> bytes:
    
    minio_client = get_storage_client()
    postgres_client = get_postgres_client()
    
    video_artifact = await postgres_client.get_artifact(
            artifact_id=video_id
    )
    if video_artifact is None:
        raise ValueError(f"Video artifact not found for ID={video_id}")

    bucket, object_name = extract_s3_minio_url(video_artifact.minio_url)

    video_bytes = minio_client.get_object(bucket=bucket, object_name=object_name)
    if video_bytes is None:
        raise ValueError(
            f"Could not fetch video from Minio: bucket={bucket}, object={object_name}"
        )

    container = av.open(video_bytes)

    fps = video_artifact.artifact_metadata['fps']
    frame_index = int(timecode_to_frame(timestamp, fps))

    timestamp_sec = frame_index / fps
    timestamp_hms = (
        f"{int(timestamp_sec // 3600):02}:"
        f"{int((timestamp_sec % 3600) // 60):02}:"
        f"{timestamp_sec % 60:06.3f}"
    )

    stream = next(s for s in container.streams if s.type == "video")
    stream.seek(frame_index) #type:ignore

    decoded_frame = None
    for frame in container.decode(stream): #type:ignore
        decoded_frame = frame
        break

    if decoded_frame is None:
        container.close()
        raise RuntimeError(f"Failed to decode frame at index={frame_index}")

    rgb_frame = decoded_frame.to_ndarray(format="rgb24") #type:ignore
    container.close()

    buf = io.BytesIO()
    Image.fromarray(rgb_frame).save(buf, format="WEBP")
    buf.seek(0)

    return buf.getvalue()
    
