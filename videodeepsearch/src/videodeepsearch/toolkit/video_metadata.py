from __future__ import annotations
from datetime import datetime
from typing import Any, Literal
from agno.tools import Toolkit, tool
from agno.tools.function import ToolResult
from loguru import logger

from videodeepsearch.clients.storage.postgre import PostgresClient
from videodeepsearch.clients.storage.minio import MinioStorageClient


class VideoInfo:
    def __init__(
        self,
        video_id: str,
        user_id: str,
        filename: str,
        duration: str,
        fps: float,
        resolution: str,
        created_at: datetime,
        minio_url: str,
        extension: str,
    ):
        self.video_id = video_id
        self.user_id = user_id
        self.filename = filename
        self.duration = duration
        self.fps = fps
        self.resolution = resolution
        self.created_at = created_at
        self.minio_url = minio_url
        self.extension = extension

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "user_id": self.user_id,
            "filename": self.filename,
            "duration": self.duration,
            "fps": self.fps,
            "resolution": self.resolution,
            "created_at": self.created_at.isoformat(),
            "minio_url": self.minio_url,
            "extension": self.extension,
        }

    def brief_repr(self) -> str:
        return (
            f"[{self.video_id}] {self.filename} | "
            f"Duration: {self.duration} | FPS: {self.fps:.2f} | "
            f"Resolution: {self.resolution}"
        )

    def detailed_repr(self) -> str:
        return (
            f"Video ID: {self.video_id}\n"
            f"  Filename: {self.filename}\n"
            f"  Duration: {self.duration}\n"
            f"  FPS: {self.fps:.2f}\n"
            f"  Resolution: {self.resolution}\n"
            f"  Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  MinIO URL: {self.minio_url}"
        )

class VideoMetadataToolkit(Toolkit):
    def __init__(
        self,
        postgres_client: PostgresClient,
        minio_client: MinioStorageClient,
    ):
        self.postgres = postgres_client
        self.storage = minio_client
        super().__init__(
            name="Video Metadata Tools",
            tools=[
                self.list_user_videos,
                self.get_video_metadata,
                self.get_video_timeline,
            ],
        )

    async def _get_video_artifacts(
        self,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[VideoInfo]:
        videos = []

        async with self.postgres.get_session() as session:
            from sqlalchemy import select
            from videodeepsearch.clients.storage.postgre.schema import ArtifactSchema

            query = select(ArtifactSchema).where(
                ArtifactSchema.artifact_type == "VideoArtifact"
            )

            if user_id:
                query = query.where(ArtifactSchema.user_id == user_id)

            query = query.order_by(ArtifactSchema.created_at.desc())
            query = query.limit(limit).offset(offset)

            result = await session.execute(query)
            artifacts = result.scalars().all()

            for artifact in artifacts:
                metadata = artifact.artifact_metadata or {}

                video_info = VideoInfo(
                    video_id=artifact.artifact_id,
                    user_id=artifact.user_id,
                    filename=metadata.get("filename", "unknown"),
                    duration=metadata.get("duration", "00:00:00"),
                    fps=metadata.get("fps", 30.0),
                    resolution=metadata.get("resolution", "unknown"),
                    created_at=artifact.created_at,
                    minio_url=artifact.minio_url,
                    extension=metadata.get("extension", "mp4"),
                )
                videos.append(video_info)

        return videos

    async def _get_all_children_artifact_types(
        self,
        video_id: str,
    ) -> set[str]:
        children = await self.postgres.get_children_artifact(video_id)
        return {child.artifact_type for child in children}


    @tool(
        description=(
            "List all videos for a user with basic metadata. "
            "Returns video IDs, filenames, duration, fps, and resolution. "
            "Use this to discover what videos are available for search.\n\n"
            "Typical workflow - Discovery phase (START HERE):\n"
            "  1. This tool - find available videos for the user\n"
            "  2. get_video_metadata - get details on a specific video\n"
            "  3. get_video_timeline - understand video structure\n"
            "  4. search.get_* tools - search for content in the videos\n\n"
            "When to use:\n"
            "  - Starting a new search session (find video IDs)\n"
            "  - User wants to see what videos they have\n"
            "  - Need to identify which videos to search\n\n"
            "Related tools:\n"
            "  - get_video_metadata: Get detailed info on a specific video\n"
            "  - get_video_timeline: See temporal structure of a video\n"
            "  - search.get_images_from_qwenvl_query: Search for visual content\n"
            "  - search.get_segments_from_event_query_mmbert: Search for events\n\n"
            "Args:\n"
            "  user_id (str): User ID to list videos for (REQUIRED)\n"
            "  limit (int): Maximum number of videos to return (default 50)\n"
            "  offset (int): Offset for pagination (default 0)"
        ),
        instructions=(
            "Use when: user wants to see available videos, "
            "need to find video IDs for further operations, "
            "starting a new search session.\n\n"
            "Best paired with: get_video_metadata, get_video_timeline (drill down into specific videos). "
            "Follow up with: search tools to find content within the videos."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def list_user_videos(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> ToolResult:
        try:
            videos = await self._get_video_artifacts(
                user_id=user_id,
                limit=limit,
                offset=offset,
            )

            if not videos:
                return ToolResult(
                    content=f"No videos found for user '{user_id}'."
                )

            lines = [
                f"Found {len(videos)} video(s) for user '{user_id}':",
                "",
            ]

            for i, video in enumerate(videos):
                lines.append(f"{i + 1}. {video.brief_repr()}")

            lines.append("")
            lines.append(f"Showing {len(videos)} results (offset: {offset}, limit: {limit})")

            return ToolResult(content="\n".join(lines))

        except Exception as e:
            logger.error(f"[VideoMetadataToolkit] list_user_videos failed: {e}")
            return ToolResult(content=f"Error: Failed to list videos - {str(e)}")

    @tool(
        description=(
            "Get detailed metadata for a specific video. "
            "Returns comprehensive information including duration, fps, resolution, "
            "file location, and creation timestamp.\n\n"
            "Typical workflow - Video inspection:\n"
            "  1. list_user_videos - find video ID\n"
            "  2. This tool - get detailed metadata\n"
            "  3. get_video_timeline - see temporal structure\n"
            "  4. search tools - find content in the video\n\n"
            "When to use:\n"
            "  - Need detailed info about a specific video\n"
            "  - Verify video properties before processing\n"
            "  - Check video existence\n\n"
            "Related tools:\n"
            "  - list_user_videos: List all videos for a user\n"
            "  - get_video_timeline: Get temporal structure\n"
            "  - search.get_images_from_qwenvl_query: Search for visual content\n"
            "Args:\n"
            "  video_id (str): Video ID to get metadata for (REQUIRED)"
        ),
        instructions=(
            "Use when: need detailed info about a specific video, "
            "want to verify video properties before processing, "
            "checking video existence.\n\n"
            "Best paired with: list_user_videos (find videos), get_video_timeline (see structure). "
            "Follow up with: search tools to find content within this video."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_video_metadata(
        self,
        video_id: str,
    ) -> ToolResult:
        try:
            artifact = await self.postgres.get_artifact(video_id)

            if artifact is None:
                return ToolResult(
                    content=f"Error: Video '{video_id}' not found."
                )

            if artifact.artifact_type != "VideoArtifact":
                return ToolResult(
                    content=f"Error: Artifact '{video_id}' is not a video (type: {artifact.artifact_type})."
                )

            metadata = artifact.artifact_metadata or {}

            lines = [
                "=== Video Metadata ===",
                "",
                f"Video ID: {video_id}",
                f"Filename: {metadata.get('filename', 'unknown')}",
                f"User ID: {artifact.user_id}",
                f"Duration: {metadata.get('duration', 'unknown')}",
                f"FPS: {metadata.get('fps', 'unknown')}",
                f"Resolution: {metadata.get('resolution', 'unknown')}",
                f"Extension: {metadata.get('extension', 'unknown')}",
                f"Created: {artifact.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "=== Full Metadata ===",
            ]

            for key, value in sorted(metadata.items()):
                lines.append(f"  {key}: {value}")

            return ToolResult(content="\n".join(lines))

        except Exception as e:
            logger.error(f"[VideoMetadataToolkit] get_video_metadata failed: {e}")
            return ToolResult(content=f"Error: Failed to get video metadata - {str(e)}")

    @tool(
        description=(
            "Get a visual timeline of video segments with timestamps and captions. "
            "Shows the temporal structure of the video with segment boundaries. "
            "Use this to navigate through video content chronologically.\n\n"
            "Typical workflow - Video structure exploration:\n"
            "  1. list_user_videos - find video ID\n"
            "  2. get_video_metadata - get video details\n"
            "  3. This tool - see temporal structure and segment boundaries\n"
            "  4. search tools or utility tools - find specific content\n\n"
            "Granularity options:\n"
            "  - 'segment': Multi-frame sequences with captions (default, most detailed)\n"
            "  - 'shot': Shot boundaries detected by scene changes\n"
            "  - 'minute': Content grouped by minute (for long videos)\n\n"
            "When to use:\n"
            "  - User wants to see video structure\n"
            "  - Need to find specific time ranges\n"
            "  - Planning temporal navigation\n"
            "  - Understanding video content distribution\n\n"
            "Related tools:\n"
            "  - list_user_videos: Find videos\n"
            "  - get_video_metadata: Get video details\n"
            "  - utility.get_adjacent_segments: Navigate between segments\n"
            "  - search.get_segments_from_event_query_mmbert: Search for events\n\n"
            "Args:\n"
            "  video_id (str): Video ID to generate timeline for (REQUIRED)\n"
            "  granularity (str): Timeline granularity - 'segment', 'shot', or 'minute' (default 'segment')"
        ),
        instructions=(
            "Use when: user wants to see video structure, "
            "need to find specific time ranges, "
            "planning temporal navigation.\n\n"
            "Best paired with: list_user_videos, get_video_metadata (find and inspect video first). "
            "Follow up with: search tools or utility.get_adjacent_segments for navigation."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_video_timeline(
        self,
        video_id: str,
        granularity: Literal["segment", "shot", "minute"] = "segment",
    ) -> ToolResult:
        try:
            video_artifact = await self.postgres.get_artifact(video_id)
            if video_artifact is None:
                return ToolResult(
                    content=f"Error: Video '{video_id}' not found."
                )

            video_metadata = video_artifact.artifact_metadata or {}
            video_fps = video_metadata.get("fps", 30.0)
            video_duration = video_metadata.get("duration", "00:00:00.000")

            lines = [
                f"=== Video Timeline for {video_id} ===",
                f"Duration: {video_duration} | FPS: {video_fps}",
                "",
            ]

            if granularity == "segment":
                segment_artifacts = await self.postgres.get_children_artifact(
                    artifact_id=video_id,
                    filter_artifact_type=["SegmentCaptionArtifact"],
                )

                if not segment_artifacts:
                    lines.append("No segments found. Video may still be processing.")
                    return ToolResult(content="\n".join(lines))

                segments_data = []
                for artifact in segment_artifacts:
                    metadata: dict = artifact.artifact_metadata or {}

                    segments_data.append({
                        "start_frame": metadata.get("start_frame", 0),
                        "end_frame": metadata.get("end_frame", 0),
                        "start_time": metadata.get("start_timestamp", "00:00:00.000"),
                        "end_time": metadata.get("end_timestamp", "00:00:00.000"),
                        "summary_caption": metadata.get("summary_caption", "")[:100],
                    })

                segments_data.sort(key=lambda x: x["start_frame"])

                lines.append(f"{'Time Range':<25} | {'Frames':<15} | Caption")
                lines.append("-" * 80)

                for seg in segments_data:
                    time_range = f"{seg['start_time']} - {seg['end_time']}"
                    frame_range = f"{seg['start_frame']} - {seg['end_frame']}"
                    caption = seg['summary_caption'][:40] + "..." if len(seg['summary_caption']) > 40 else seg['summary_caption']

                    lines.append(f"{time_range:<25} | {frame_range:<15} | {caption}")

                lines.append("")
                lines.append(f"Total segments: {len(segments_data)}")

            elif granularity == "shot":
                autoshot_artifacts = await self.postgres.get_children_artifact(
                    artifact_id=video_id,
                    filter_artifact_type=["AutoshotArtifact"],
                )
                
                if not autoshot_artifacts:
                    lines.append("No shot boundaries found. Video may still be processing.")
                    return ToolResult(content="\n".join(lines))

                for artifact in autoshot_artifacts:
                    metadata: dict = artifact.artifact_metadata or {}
                    shots = metadata.get("segments", [])

                    if shots:
                        lines.append(f"Total shots detected: {len(shots)}")
                        lines.append("")
                        lines.append(f"{'Shot #':<8} | {'Start Frame':<12} | {'End Frame':<12}")
                        lines.append("-" * 40)

                        for i, shot in enumerate(shots[:50]):  
                            start_frame, end_frame = shot
                            lines.append(
                                f"{i + 1:<8} | {start_frame:<12} | "
                                f"{end_frame:<12}"
                            )

                        if len(shots) > 50:
                            lines.append(f"... and {len(shots) - 50} more shots")

                        break

            elif granularity == "minute":
                segment_artifacts = await self.postgres.get_children_artifact(
                    artifact_id=video_id,
                    filter_artifact_type=["SegmentCaptionArtifact"],
                )

                if not segment_artifacts:
                    lines.append("No segments found. Video may still be processing.")
                    return ToolResult(content="\n".join(lines))

                segments_data = []
                for artifact in segment_artifacts:
                    metadata: dict = artifact.artifact_metadata or {}

                    start_sec = metadata.get("start_sec", 0)
                    segments_data.append({
                        "start_sec": start_sec,
                        "minute": int(start_sec // 60),
                        "summary_caption": metadata.get("summary_caption", "")[:100],
                    })

                
                from collections import defaultdict
                minute_groups = defaultdict(list)
                for seg in segments_data:
                    minute_groups[seg["minute"]].append(seg)

                lines.append(f"{'Minute':<8} | {'Num of Segments':<10} | Sample Caption")
                lines.append("-" * 60)

                for minute in sorted(minute_groups.keys()):
                    segments = minute_groups[minute]
                    sample = segments[0]["summary_caption"][:100] + "..." if segments[0]["summary_caption"] else "N/A"
                    lines.append(f"{minute:<8} | {len(segments):<10} | {sample}")

                lines.append("")
                lines.append(f"Total minutes with content: {len(minute_groups)}")

            return ToolResult(content="\n".join(lines))

        except Exception as e:
            logger.error(f"[VideoMetadataToolkit] get_video_timeline failed: {e}")
            return ToolResult(content=f"Error: Failed to get video timeline - {str(e)}")

__all__ = ["VideoMetadataToolkit"]