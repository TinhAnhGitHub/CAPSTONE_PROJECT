"""
This file contains the tools for agents to simulate the human-behaviour to interfact with the video, scanning images...
"""
from agentic_ai.tools.schema.artifact import ImageObjectInterface, SegmentObjectInterface
from agentic_ai.tools.clients.postgre.client import PostgresClient
from agentic_ai.tools.clients.minio.client import StorageClient
from ingestion.core.artifact.schema import SegmentCaptionArtifact, ImageCaptionArtifact
from typing import Annotated, Literal, cast
from .helper import extract_s3_minio_url
from .registry import tool_registry

##########################
# Segment-based operations
##########################



@tool_registry.register(
    category='video/image/segment interaction',
    tags=["interaction","segment"],
    dependencies=[
        "postgres_client",
        "minio_client",
    ]
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
    children_segments = await postgres_client.get_children_artifact(artifact_id=parent_video_id, filter_artifact_type=[SegmentCaptionArtifact.__name__])

    filter_segments: list[SegmentObjectInterface] = []
    for child in children_segments:
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
                        segment_caption_query=None
                    )
                )
        
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
                        segment_caption_query=None
                    )
                )
        
    return filter_segments[:hop] if include_within_range else filter_segments[hop-1]

##########################
# Image-based operations
##########################

@tool_registry.register(
    category='video/image/segment interaction',
    tags=["intefaction", "image"],
    dependencies=[
        "postgres_client",
        "minio_client",
    ]
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

    filter_segments = []

    for child in child_segments:
        minio_path = child.minio_url
        bucket, object_name = extract_s3_minio_url(minio_path)
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
                        query=None
                    )
                )
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
                        query=None
                    )
                )
    return filter_segments[:hop] if include_within_range else filter_segments[hop-1]
    
