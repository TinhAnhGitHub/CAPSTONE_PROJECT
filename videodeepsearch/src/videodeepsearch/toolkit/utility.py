from __future__ import annotations

import base64
import io
from typing import Any, Literal

import av
import cv2
from agno.media import Image as AgnoImage
from agno.tools import Toolkit, tool
from agno.tools.function import ToolResult
from loguru import logger
from PIL import Image as PILImage
from tqdm import tqdm

from videodeepsearch.clients.storage.minio import MinioStorageClient
from videodeepsearch.clients.storage.postgre import PostgresClient
from videodeepsearch.schemas import ImageInterface, SegmentInterface
from videodeepsearch.toolkit.common import (
    extract_s3_minio_url,
    time_to_seconds,
    parse_time_safe,
    time_range_overlap,
    convert_time_to_frame,
    timecode_to_frame,
)


def format_interface_list(items: list[Any], item_type: str) -> str:
    """Format a list of interfaces for ToolResult content."""
    if not items:
        return f"No {item_type}s found."

    lines = [f"Found {len(items)} {item_type}(s):", ""]
    for i, item in enumerate(items):
        lines.append(f"[{i}] {item.detailed_representation()}")

    return "\n".join(lines)



ASR_TOKEN_TEMPLATE = """
Start time/index: {start}/{start_frame}
End time/index:   {end}/{end_frame}
ASR content:      {text}
"""

SNIPPET_TEMPLATE = """
ASR transcript context around the segment:

▶ Segment range: {segment_start_time} → {segment_end_time}
▶ Frame range: {segment_start_frame} → {segment_end_frame}
▶ Context window: ±{window_seconds} seconds

--------------------- TRANSCRIPT CONTEXT ---------------------
{context}
--------------------------------------------------------------

Note: Some ASR lines may include adjacent context beyond the target segment.
Focus on lines semantically aligned with the segment's content.
"""

IMAGE_ASR_TEMPLATE = """
Here are the ASR captured around the image at timestamp/frame_index: {image_timestamp}/{image_frame_index}
around the window seconds: {window_seconds}

{context}

The ASR might have irrelevant events/context, so just focus on the related ASR segments!
"""


class UtilityToolkit(Toolkit):
    """Toolkit for ASR context retrieval and video navigation.

    Provides tools for:
    - Retrieving ASR transcript context around segments/images
    - Navigating to adjacent video segments/frames (temporal exploration)
    - Extracting raw frames from videos for visual inspection

    All tools return ToolResult for unified interface.
    """

    def __init__(
        self,
        postgres_client: PostgresClient,
        minio_client: MinioStorageClient,
    ):
        """Initialize the UtilityToolkit.

        Args:
            postgres_client: PostgreSQL client for artifact metadata
            minio_client: MinIO storage client for ASR data and videos
        """
        self.postgres = postgres_client
        self.storage = minio_client
        super().__init__(
            name="Utility Tools",
            tools=[
                self.get_related_asr_from_segment,
                self.get_related_asr_from_image,
                self.get_adjacent_segments,
                self.get_adjacent_images,
            ],
        )
        
    @tool(
        description=(
            "Retrieve spoken words (ASR transcript) around a video segment for context. "
            "Extracts the audio transcript within a time window surrounding a segment, "
            "providing spoken context to verify or enrich visual findings.\n\n"
            "Typical workflow - Use AFTER any videosearch tool that finds a segment:\n"
            "  1. search.get_segments_from_event_query_mmbert - find matching segments\n"
            "  2. Or search.get_segments_from_qwenvl_query - find matching segments\n"
            "  3. This tool - get spoken context around found segment\n"
            "Related tools:\n"
            "  - get_related_asr_from_image: For ASR context around a specific image/frame\n"
            "  - get_adjacent_segments: For temporal navigation to neighboring segments\n"
            "Args:\n"
            "  video_id (str): Video ID (REQUIRED)\n"
            "  segment_start_time (str): Segment start time in HH:MM:SS.sss format (REQUIRED)\n"
            "  segment_end_time (str): Segment end time in HH:MM:SS.sss format (REQUIRED)\n"
            "  window_seconds (float): Time window around segment in ±seconds (default 10.0)"
        ),
        instructions=(
            "Use when: found a promising segment but need verbal context, "
            "user query mentioned dialogue/speech/audio events, "
            "want to verify if visual match aligns with spoken content.\n\n"
            "Best paired with: search.get_segments_from_event_query_mmbert, search.get_segments_from_qwenvl_query (find segments first). "
        ),
    )
    async def get_related_asr_from_segment(
        self,
        video_id: str,
        segment_start_time: str,
        segment_end_time: str,
        window_seconds: float = 10.0,
    ) -> ToolResult:
        """Retrieve ASR transcript context around a video segment.

        Args:
            video_id: Video ID
            segment_start_time: Segment start time (HH:MM:SS.sss)
            segment_end_time: Segment end time (HH:MM:SS.sss)
            window_seconds: Time window around segment (±seconds, default 10)

        Returns:
            ToolResult with ASR transcript context
        """
        video_artifact = await self.postgres.get_artifact(artifact_id=video_id)
        if video_artifact is None:
            return ToolResult(content=f"Error: Video ID {video_id} does not exist")

        video_fps = video_artifact.artifact_metadata.get("fps", 30.0)
        duration = video_artifact.artifact_metadata.get("duration", "00:00:00.000")

        asr_artifacts = await self.postgres.get_children_artifact(
            artifact_id=video_id,
            filter_artifact_type=["ASRArtifact"],
        )
        print(f"asr_artifacts: {asr_artifacts[:2]}")

        if not asr_artifacts:
            return ToolResult(content=f"Error: No ASR data found for video {video_id}")

        segment_start = time_to_seconds(segment_start_time)
        segment_end = time_to_seconds(segment_end_time)
        video_duration = time_to_seconds(duration)

        if segment_start < 0:
            return ToolResult(content=f"Error: start_time must be >= 0, got: {segment_start_time}")

        if segment_end > video_duration:
            return ToolResult(content=f"Error: end_time ({segment_end_time}) exceeds video duration ({duration})")

        if segment_start >= segment_end:
            return ToolResult(content=f"Error: start_time ({segment_start_time}) must be < end_time ({segment_end_time})")

        window_start = segment_start - window_seconds
        window_end = segment_end + window_seconds

        matching_asr = []

        for artifact in asr_artifacts:
            metadata: dict = artifact.artifact_metadata
            start_time, end_time = metadata['timestamp']
            start_sec = time_to_seconds(start_time)
            end_sec = time_to_seconds(end_time)

            if time_range_overlap(window_start, window_end, start_sec, end_sec):
                text = metadata['text']
                start_frame, end_frame = metadata['frame_num']
                matching_asr.append((start_sec, start_time, start_frame, end_time, end_frame, text))

        matching_asr.sort(key=lambda x: x[0])

        context_parts = []
        for _, start_time, start_frame, end_time, end_frame, text in matching_asr:
            context_parts.append(ASR_TOKEN_TEMPLATE.format(
                start=start_time,
                start_frame=start_frame,
                end=end_time,
                end_frame=end_frame,
                text=text,
            ).strip())


        return ToolResult(content=SNIPPET_TEMPLATE.format(
            segment_start_time=segment_start_time,
            segment_end_time=segment_end_time,
            segment_start_frame=convert_time_to_frame(segment_start_time, video_fps),
            segment_end_frame=convert_time_to_frame(segment_end_time, video_fps),
            window_seconds=window_seconds,
            context="\n\n".join(context_parts),
        ))

    @tool(
        description=(
            "Retrieve spoken words (ASR transcript) around a specific image/frame. "
            "Extracts audio transcript within a time window surrounding an image's timestamp.\n\n"
            "Typical workflow - Use AFTER any image search tool:\n"
            "  1. search.get_images_from_caption_query_mmbert - find matching images\n"
            "  2. Or search.get_images_from_qwenvl_query - find matching images\n"
            "  3. This tool - get ASR context around the found image/frame\n"
            "Related tools:\n"
            "  - get_related_asr_from_segment: For ASR context around a video segment\n"
            "  - get_adjacent_images: For temporal navigation to neighboring frames\n"
            "Args:\n"
            "  video_id (str): Video ID (REQUIRED)\n"
            "  image_timestamp (str): Image timestamp in HH:MM:SS.sss format (REQUIRED)\n"
            "  window_seconds (float): Time window around image in ±seconds (default 10.0)"
        ),
        instructions=(
            "Use when: found a promising image but need verbal context around that moment, "
            "user query mentioned dialogue at specific frames.\n\n"
            "Best paired with: search.get_images_from_caption_query_mmbert, search.get_images_from_qwenvl_query (find images first). "
        ),
    )
    async def get_related_asr_from_image(
        self,
        video_id: str,
        image_timestamp: str,
        window_seconds: float = 10.0,
    ) -> ToolResult:
        """Retrieve ASR transcript context around an image/frame.

        Args:
            video_id: Video ID
            image_timestamp: Image timestamp (HH:MM:SS.sss)
            window_seconds: Time window around image (±seconds, default 10)

        Returns:
            ToolResult with ASR transcript context
        """
        video_artifact = await self.postgres.get_artifact(artifact_id=video_id)
        if video_artifact is None:
            return ToolResult(content=f"Error: Video ID {video_id} does not exist")

        video_fps = video_artifact.artifact_metadata.get("fps", 30.0)

        asr_artifacts = await self.postgres.get_children_artifact(
            artifact_id=video_id,
            filter_artifact_type=["ASRArtifact"],
        )

        if not asr_artifacts:
            return ToolResult(content=f"Error: No ASR data found for video {video_id}")

        ts_center = time_to_seconds(image_timestamp)
        window_start = ts_center - window_seconds
        window_end = ts_center + window_seconds

        # Collect matching ASR entries with their start time for sorting
        matching_asr = []

        for artifact in asr_artifacts:
            metadata: dict = artifact.artifact_metadata
            start_time, end_time = metadata['timestamp']
            start_sec = time_to_seconds(start_time)
            end_sec = time_to_seconds(end_time)

            if time_range_overlap(window_start, window_end, start_sec, end_sec):
                text = metadata['text']
                start_frame, end_frame = metadata['frame_num']
                matching_asr.append((start_sec, start_time, start_frame, end_time, end_frame, text))

        # Sort by start time
        matching_asr.sort(key=lambda x: x[0])

        context_parts = []
        for _, start_time, start_frame, end_time, end_frame, text in matching_asr:
            context_parts.append(ASR_TOKEN_TEMPLATE.format(
                start=start_time,
                start_frame=start_frame,
                end=end_time,
                end_frame=end_frame,
                text=text,
            ).strip())

        frame_index = convert_time_to_frame(image_timestamp, video_fps)

        return ToolResult(content=IMAGE_ASR_TEMPLATE.format(
            image_timestamp=image_timestamp,
            image_frame_index=frame_index,
            window_seconds=window_seconds,
            context="\n\n".join(context_parts),
        ))

    @tool(
        description=(
            "Navigate to adjacent segments before/after a reference segment. "
            "Retrieves surrounding video segments relative to a known segment, "
            "enabling temporal exploration of video content. Like 'turning pages' in a video book.\n\n"
            "Typical workflow - Use for temporal exploration:\n"
            "  1. search.get_segments_from_event_query_mmbert - find initial matching segment\n"
            "  2. Or search.get_segments_from_qwenvl_query - find initial matching segment\n"
            "  3. This tool - explore adjacent segments (forward or backward)\n"
            "  4. utility.get_related_asr_from_segment - get context for each adjacent segment\n\n"
            "Temporal navigation patterns:\n"
            "  - 'forward' + hop=1: Next segment after current (like turning page forward)\n"
            "  - 'backward' + hop=3: Previous 3 segments (likerewinding for context)\n"
            "  - include_range=True: Returns all segments within hop range\n"
            "  - include_range=False: Returns only the segment exactly at hop distance\n\n"
            "Related tools:\n"
            "  - get_adjacent_images: For frame-by-frame navigation (finer granularity)\n"
            "  - get_related_asr_from_segment: Get spoken context for any segment\n"
            "  - video.get_video_timeline: Get overview of all segment boundaries\n\n"
            "Args:\n"
            "  video_id (str): Video ID (REQUIRED)\n"
            "  pivot_start_frame (int): Reference segment start frame (REQUIRED)\n"
            "  pivot_end_frame (int): Reference segment end frame (REQUIRED)\n"
            "  hop (int): Number of segments to hop (default 1)\n"
            "  direction (str): 'forward' or 'backward' navigation (default 'forward')\n"
            "  include_range (bool): If True, include all segments within hop range (default True)"
        ),
        instructions=(
            "Use when: found a promising segment and want to see what happens before/after, "
            "need temporal context around a matching segment, "
            "verifying if an event continues across multiple segments.\n\n"
            "Best paired with: search.get_segments_from_event_query_mmbert, search.get_segments_from_qwenvl_query (find base segment first). "
            "Follow up with: get_related_asr_from_segment for context on each adjacent segment."
        ),
    )
    async def get_adjacent_segments(
        self,
        video_id: str,
        pivot_start_frame: int,
        pivot_end_frame: int,
        hop: int = 1,
        direction: Literal["forward", "backward"] = "forward",
        include_range: bool = True,
    ) -> ToolResult:
        """Get adjacent video segments for temporal navigation.

        Args:
            video_id: Video ID
            pivot_start_frame: Reference segment start frame
            pivot_end_frame: Reference segment end frame
            hop: Number of segments to hop (default 1)
            direction: 'forward' or 'backward' navigation
            include_range: If True, include all segments within hop range

        Returns:
            ToolResult with adjacent segments
        """
        segment_artifacts = await self.postgres.get_children_artifact(
            artifact_id=video_id,
            filter_artifact_type=["SegmentCaptionArtifact"],
        )
        print(segment_artifacts[0])
        segments: list[SegmentInterface] = []

        for artifact in segment_artifacts:
            start_frame = artifact.artifact_metadata["start_frame"]
            end_frame = artifact.artifact_metadata["end_frame"]
            segment_inferface = SegmentInterface(
                id=str(artifact.artifact_id),
                related_video_id=video_id,
                user_bucket=artifact.user_id,
                start_frame=artifact.artifact_metadata.get("start_frame", 0),
                end_frame=artifact.artifact_metadata.get("end_frame", 0),
                start_time=artifact.artifact_metadata.get("start_timestamp", "00:00:00.000"),
                end_time=artifact.artifact_metadata.get("end_timestamp", "00:00:00.000"),
                segment_caption=artifact.artifact_metadata.get("summary_caption", ""),
                score=0.0,
                
            )

            if direction == "forward":
                if start_frame >= pivot_end_frame: 
                    segments.append(segment_inferface)
            else:
                if end_frame <= pivot_start_frame:
                    segments.append(segment_inferface)

        segments.sort(
            key=lambda s: parse_time_safe(s.end_time),
            reverse=(direction == "backward"),
        )

        if include_range:
            result_segments = segments[:hop]
        else:
            if len(segments) >= hop:
                result_segments = [segments[hop - 1]]
            else:
                result_segments = segments[:1] if segments else []

        return ToolResult(content=format_interface_list(result_segments, "segment"))

    @tool(
        description=(
            "Navigate to adjacent frames before/after a reference image. "
            "Retrieves surrounding frames relative to a known image, enabling frame-by-frame "
            "exploration. Like stepping through video one frame at a time.\n\n"
            "Typical workflow - Use for frame-level temporal exploration:\n"
            "  1. search.get_images_from_caption_query_mmbert - find initial matching image\n"
            "  2. Or search.get_images_from_qwenvl_query - find initial matching image\n"
            "  3. This tool - explore adjacent frames (forward or backward)\n"
            "  4. utility.get_related_asr_from_image - get ASR context for each frame\n\n"
            "Temporal navigation patterns:\n"
            "  - 'forward' + hop=1: Next frame after current (for frame-by-frame progression)\n"
            "  - 'backward' + hop=5: Previous 5 frames (for quick context回顾)\n"
            "  - include_range=True: Returns all frames within hop range\n"
            "  - include_range=False: Returns only the frame exactly at hop distance\n\n"
            "Related tools:\n"
            "  - get_adjacent_segments: For segment-level navigation (coarser granularity)\n"
            "  - get_related_asr_from_image: Get ASR context for any image/frame\n"
            "Args:\n"
            "  video_id (str): Video ID (REQUIRED)\n"
            "  image_frame_index (int): Reference image frame index (REQUIRED)\n"
            "  hop (int): Number of images to hop (default 1)\n"
            "  direction (str): 'forward' or 'backward' navigation (default 'forward')\n"
            "  include_range (bool): If True, include all images within hop range (default True)"
        ),
        instructions=(
            "Use when: found a good frame and want to see adjacent frames, "
            "checking for motion/action across consecutive frames, "
            "verifying object persistence across time.\n\n"
            "Best paired with: search.get_images_from_caption_query_mmbert, search.get_images_from_qwenvl_query (find base image first). "
            "Follow up with: get_related_asr_from_image for ASR context on each adjacent frame."
        ),
    )
    async def get_adjacent_images(
        self,
        video_id: str,
        image_frame_index: int,
        hop: int = 1,
        direction: Literal["forward", "backward"] = "forward",
        include_range: bool = True,
    ) -> ToolResult:
        """Get adjacent images/frames for temporal navigation.

        Args:
            video_id: Video ID
            image_frame_index: Reference image frame index
            hop: Number of images to hop (default 1)
            direction: 'forward' or 'backward' navigation
            include_range: If True, include all images within hop range

        Returns:
            ToolResult with adjacent images
        """
        image_artifacts = await self.postgres.get_children_artifact(
            artifact_id=video_id,
            filter_artifact_type=["ImageCaptionArtifact"],
        )

        images: list[ImageInterface] = []

        for artifact in image_artifacts:
            metadata: dict = artifact.artifact_metadata

            image = ImageInterface(
                id=str(artifact.artifact_id),
                related_video_id=video_id,
                user_bucket=artifact.user_id,
                frame_index=metadata.get("frame_index", 0),
                timestamp=metadata.get("timestamp", "00:00:00.000"),
                image_caption=metadata.get("caption", ""),
                score=0.0,
                timestamp_sec=metadata.get("timestamp_sec"),
                related_video_fps=metadata.get("fps"),
            )

            if direction == "forward":
                if image.frame_index >= image_frame_index:
                    images.append(image)
            else:
                if image.frame_index <= image_frame_index:
                    images.append(image)

        images.sort(
            key=lambda i: i.frame_index,
            reverse=(direction == "backward"),
        )

        if include_range:
            result_images = images[:hop]
        else:
            if len(images) >= hop:
                result_images = [images[hop - 1]]
            else:
                result_images = images[:1] if images else []

        return ToolResult(content=format_interface_list(result_images, "image"))

    


__all__ = ["UtilityToolkit"]