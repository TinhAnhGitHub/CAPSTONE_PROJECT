"""
This file contains the tools for agents to simulate the human-behaviour to interfact with the video, scanning images...
"""
from videodeepsearch.tools.schema.artifact import ImageObjectInterface, SegmentObjectInterface, VideoInterface
import asyncio
import io
from videodeepsearch.tools.clients.postgre.client import PostgresClient
from videodeepsearch.tools.clients.minio.client import StorageClient
from ingestion.prefect_agent.service_asr.core.schema import  ASRResult
from ingestion.core.artifact.schema import SegmentCaptionArtifact, ImageCaptionArtifact, ASRArtifact
from typing import Annotated, Literal, cast
import cv2


from .helper import extract_s3_minio_url, create_tmp_file_from_minio_object, timecode_to_frame, parse_time_safe
from .registry import tool_registry

##########################
# Segment-based operations
##########################


@tool_registry.register(
    category="Interaction/Video",
    tags=["video", "metadata", "context"],
    dependencies=["postgres_client"]
)
async def get_video_from_segment(
    segment_interface: SegmentObjectInterface,
    postgres_client: PostgresClient,
) -> VideoInterface:
    related_video_id = segment_interface.related_video_id
    video_artifact_metadata = await postgres_client.get_artifact(artifact_id=related_video_id)
    if video_artifact_metadata is None:
        raise ValueError()
    metadata = video_artifact_metadata.artifact_metadata
    return VideoInterface(
        video_id=related_video_id,
        fps=metadata['fps'],
        minio_path=video_artifact_metadata.minio_url,
        duration=metadata['duration']
    )

@tool_registry.register(
    category="Interaction/Video",
    tags=["video", "metadata", "context"],
    dependencies=["postgres_client"]
)
async def get_video_from_image(
    image_interface: ImageObjectInterface,
    postgres_client: PostgresClient
) -> VideoInterface:
    related_video_id = image_interface.related_video_id
    video_artifact_metadata = await postgres_client.get_artifact(artifact_id=related_video_id)
    if video_artifact_metadata is None:
        raise ValueError()
    metadata = video_artifact_metadata.artifact_metadata
    return VideoInterface(
        video_id=related_video_id,
        fps=metadata['fps'],
        minio_path=video_artifact_metadata.minio_url,
        duration=metadata['duration']
    )


@tool_registry.register(
    category="Interaction/ASR",
    tags=["asr", "transcript", "multimodal", "video"],
    dependencies=["postgres_client", "minio_client"]
)
async def get_asr_from_video(
    video:VideoInterface,
    postgres_client: PostgresClient,
    minio_client: StorageClient
)->str:
    asr_segments = await postgres_client.get_children_artifact(
        artifact_id=video.video_id,
        filter_artifact_type=[ASRArtifact.__name__]
    )
    asr_segment = asr_segments[0]

    bucket, object_name = extract_s3_minio_url(asr_segment.minio_url)
    json_dict = minio_client.read_json(bucket=bucket, object_name=object_name)
    if json_dict is None:
        raise ValueError()


    asr_response = ASRResult.model_validate(json_dict)
    tokens = asr_response.tokens
    context = []
    for token in tokens:
        context_token = f"""
        Start time/index: {token.start}/{token.start_frame}
        End time/index:   {token.end}/{token.end_frame}
        ASR content:      {token.text}
        """
        context.append(context_token.strip())
    return "\n\n".join(context)

    


@tool_registry.register(
    category="Interaction/Segment",
    tags=["segment", "metadata", "caption", "structure"],
    dependencies=["postgres_client", "minio_client"]
)
async def get_all_segment_info_from_video_interface(
    video_interface: VideoInterface,
    postgres_client: PostgresClient,
    minio_client: StorageClient
)->list[SegmentObjectInterface]:
    child_artifacts = await postgres_client.get_children_artifact(artifact_id=video_interface.video_id, filter_artifact_type=[SegmentCaptionArtifact.__name__ ])
    print(child_artifacts)
    children_segments = list(
        filter(lambda x: x.artifact_type==SegmentCaptionArtifact.__name__, child_artifacts)
    )
    result: list[SegmentObjectInterface] = []
    for artifact in children_segments:
        minio_url = artifact.minio_url
        bucket, object_name = extract_s3_minio_url(minio_url)

        json_dict = minio_client.read_json(bucket=bucket, object_name=object_name)
        if json_dict is None:
            raise ValueError()

        result.append(
            SegmentObjectInterface(
                related_video_id=video_interface.video_id,
                start_frame_index=json_dict['start_frame'],
                end_frame_index=json_dict['end_frame'],
                start_time=json_dict['start_timestamp'],
                end_time=json_dict['end_timestamp'],
                caption_info=json_dict['caption'],
                score=None,
                minio_path=minio_url,
                segment_caption_query=None
            )
        )
    return result

@tool_registry.register(
    category="Interaction/Segment",
    tags=["segment", "navigation", "context", "video"],
    dependencies=["postgres_client", "minio_client"]
)
async def get_segments(
    current_segment: SegmentObjectInterface,
    hop: Annotated[int, "Define how many steps you want to include/skip. The range is predefined before. For example if you want to see a few segment ahead, it will be like 2 or 3"],
    include_within_range: bool,
    forward_or_backward: Literal['forward', 'backward'],
    postgres_client: PostgresClient,
    minio_client: StorageClient
) -> SegmentObjectInterface | list[SegmentObjectInterface]:
    """
    Given a current segment, return the next segments, based on the hop size
    Args:
        current_segment: 
        hop (int): the hop size
        include_within_range (bool): If True, then all the range within hop is included, else just return the destination segment
    """


    parent_video_id = current_segment.related_video_id
    children_artifact = await postgres_client.get_children_artifact(artifact_id=parent_video_id, filter_artifact_type=[SegmentCaptionArtifact.__name__])

    filter_segments: list[SegmentObjectInterface] = []
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
            if  segment_artifact.start_frame >= current_segment.end_frame_index:
                filter_segments.append(
                    SegmentObjectInterface(
                        related_video_id=segment_artifact.related_video_id,
                        start_frame_index=segment_artifact.start_frame,
                        end_frame_index=segment_artifact.end_frame,
                        caption_info=caption,
                        start_time=segment_artifact.start_timestamp,
                        end_time=segment_artifact.end_timestamp,
                        score=None,
                        minio_path=segment_artifact.minio_url_path,
                        segment_caption_query=None
                    )
                )
            filter_segments.sort(key=lambda s: parse_time_safe(s.end_time))
        
        elif forward_or_backward == 'backward':
            if segment_artifact.end_frame <= current_segment.start_frame_index:
                filter_segments.append(
                    SegmentObjectInterface(
                        related_video_id=segment_artifact.related_video_id,
                        start_frame_index=segment_artifact.start_frame,
                        end_frame_index=segment_artifact.end_frame,
                        caption_info=caption,
                        start_time=segment_artifact.start_timestamp,
                        end_time=segment_artifact.end_timestamp,
                        score=None,
                        minio_path=segment_artifact.minio_url_path,
                        segment_caption_query=None
                    )
                )
            filter_segments.sort(key=lambda s: parse_time_safe(s.end_time), reverse=True)
    
    return filter_segments[:hop] if include_within_range else filter_segments[hop-1]

##########################
# Image-based operations
##########################

@tool_registry.register(
    category="Interaction/Image",
    tags=["image", "navigation", "context", "video"],
    dependencies=["postgres_client", "minio_client"]
)
async def get_images(
    image: ImageObjectInterface,
    hop: int,
    include_within_range: bool,
    forward_or_backward: Literal['forward', 'backward'],
    postgres_client: PostgresClient,
    minio_client: StorageClient
) -> ImageObjectInterface | list[ImageObjectInterface]:


    parent_video_id = image.related_video_id
    child_segments = await postgres_client.get_children_artifact(artifact_id=parent_video_id, filter_artifact_type=[ImageCaptionArtifact.__name__])

    print(f"{child_segments=}")
    filter_segments = []

    for child in child_segments:
        minio_path = child.minio_url
        bucket, object_name = extract_s3_minio_url(minio_path)
        print(bucket)
        image_id = cast(str,child.parent_artifact_id)
        image_metadata = await postgres_client.get_artifact(artifact_id=image_id)
        if image_metadata is None:
            raise ValueError(f"The image id {image_id} should be exists")

        json_dict = cast(dict,minio_client.read_json(bucket=bucket, object_name=object_name))

        caption_image = json_dict['caption']
        del json_dict['caption']
        image_caption_artifact = ImageCaptionArtifact.model_validate(json_dict)

        if forward_or_backward == 'forward': 
            if image_caption_artifact.frame_index >= image.frame_index:
                filter_segments.append(
                    ImageObjectInterface(
                        related_video_id=parent_video_id,
                        frame_index=image_caption_artifact.frame_index,
                        caption_info=caption_image,
                        minio_path=image_metadata.minio_url,
                        timestamp=image_caption_artifact.time_stamp,
                        score=None,
                        # reference_query_image=None,
                        query=None
                    )
                )
            filter_segments.sort(key=lambda s: parse_time_safe(s.timestamp))
        elif forward_or_backward == 'backward':
            if image_caption_artifact.frame_index <= image.frame_index:
                filter_segments.append(
                    ImageObjectInterface(
                        related_video_id=parent_video_id,
                        frame_index=image_caption_artifact.frame_index,
                        caption_info=caption_image,
                        minio_path=image_metadata.minio_url,
                        timestamp=image_caption_artifact.time_stamp,
                        score=None,
                        # reference_query_image=None,
                        query=None
                    )
                )
            filter_segments.sort(key=lambda s: parse_time_safe(s.timestamp), reverse=True)
    return filter_segments[:hop] if include_within_range else filter_segments[hop-1]
    


@tool_registry.register(
    category="Interaction/FrameExtraction",
    tags=["video", "frame", "extract", "sampling"],
    dependencies=["postgres_client", "minio_client"]
)
async def extract_frames_by_time_window(
    video_interface: VideoInterface,
    start_time: Annotated[str, "Start time in HH:MM:SS.sss format"],
    end_time: Annotated[str, "End time in HH:MM:SS.sss format"],
    fps_sample_rate: Annotated[int, ""],
    
    postgres_client: PostgresClient,
    minio_client: StorageClient,
    agent_bucket: str,
    agent_object_folder: str
)->list[ImageObjectInterface]:
    video_artifact = await postgres_client.get_artifact(artifact_id=video_interface.video_id)
    if video_artifact is None:
        raise ValueError()

    bucket, object_name = extract_s3_minio_url(video_artifact.minio_url)

    video_object = minio_client.get_object(bucket=bucket, object_name=object_name)
    if video_object is None:
        raise ValueError()
    
    loop = asyncio.get_event_loop()
    video_tmp_path = await loop.run_in_executor(
        None,
        lambda: create_tmp_file_from_minio_object(
            file_bytes=video_object,
            extension=video_artifact.artifact_metadata['extension']
        )
    )
    fps = video_interface.fps
    start_frame = int(timecode_to_frame(start_time, fps))
    end_frame = int(timecode_to_frame(end_time, fps))

    cap = cv2.VideoCapture(video_tmp_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file {video_tmp_path}")

    frames_data = []
    frame_index = start_frame
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while frame_index <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_sec = frame_index / fps
        
        success, buffer = cv2.imencode('.webp', frame)
        if not success:
            frame_index += fps_sample_rate
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            continue

        
        image_bytes = io.BytesIO(buffer.tobytes())
        timestamp_sec = frame_index / fps
        timestamp_hms = f"{int(timestamp_sec // 3600):02}:{int((timestamp_sec % 3600) // 60):02}:{timestamp_sec % 60:06.3f}"

        object_save_image_name = (
            f"{agent_object_folder}/{video_interface.video_id}/frames/{frame_index}.webp"
        )
        s3_url = minio_client.upload_fileobj(
            bucket=agent_bucket,
            object_name=object_save_image_name,
            content_type="image/webp",
            file_obj=image_bytes
        )
        
        image_interface = ImageObjectInterface(
            related_video_id=video_interface.video_id,
            frame_index=frame_index,
            caption_info=None,
            minio_path=s3_url,
            timestamp=timestamp_hms,
            score=None,
            # reference_query_image=None,
            query=None
        )

        frames_data.append(image_interface)
        frame_index += fps_sample_rate
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    cap.release()
    return frames_data

    

@tool_registry.register(
    category="Interaction/FrameExtraction",
    tags=["video", "frame", "extract", "single"],
    dependencies=["postgres_client", "minio_client"]
)
async def extract_frame_time(
    video_interface: VideoInterface,
    timestamp: Annotated[str, "Start time in HH:MM:SS.sss format"],
    # lifespan dependency injection
    postgres_client: PostgresClient,
    minio_client: StorageClient,

    # Agent Run injection
    agent_bucket: str,
    agent_object_folder: str
) -> None | ImageObjectInterface:
    video_artifact = await postgres_client.get_artifact(artifact_id=video_interface.video_id)
    if video_artifact is None:
        raise ValueError()

    bucket, object_name = extract_s3_minio_url(video_artifact.minio_url)

    video_object = minio_client.get_object(bucket=bucket, object_name=object_name)
    if video_object is None:
        raise ValueError()
    
    loop = asyncio.get_event_loop()
    video_tmp_path = await loop.run_in_executor(
        None,
        lambda: create_tmp_file_from_minio_object(
            file_bytes=video_object,
            extension=video_artifact.artifact_metadata['extension']
        )
    )
    fps = video_interface.fps
    frame_index  = float(timecode_to_frame(time_str=timestamp, fps=fps))

    cap = cv2.VideoCapture(video_tmp_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file {video_tmp_path}")
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = cap.read()
    if not ret:
        return None
    
    timestamp_sec = frame_index / fps
    
    success, buffer = cv2.imencode('.webp', frame)
    if not success:
        return None

    
    image_bytes = io.BytesIO(buffer.tobytes())
    timestamp_sec = frame_index / fps
    timestamp_hms = f"{int(timestamp_sec // 3600):02}:{int((timestamp_sec % 3600) // 60):02}:{timestamp_sec % 60:06.3f}"

    object_save_image_name = (
        f"{agent_object_folder}/{video_interface.video_id}/frames/{frame_index}.webp"
    )
    s3_url = minio_client.upload_fileobj(
        bucket=agent_bucket,
        object_name=object_save_image_name,
        content_type="image/webp",
        file_obj=image_bytes
    )


    return ImageObjectInterface(
        related_video_id=video_interface.video_id,
        frame_index=int(frame_index),
        caption_info=None,
        minio_path=s3_url,
        timestamp=timestamp_hms,
        score=None,
        # reference_query_image=None,
        query=None
    )
