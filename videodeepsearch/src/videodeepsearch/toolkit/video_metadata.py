"""Video Metadata Toolkit for video-level operations.

This toolkit provides tools for:
- Listing user videos
- Retrieving video metadata
- Getting video summaries
- Generating video timelines
- Checking processing status

All tools return ToolResult for unified interface.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlparse

from agno.tools import Toolkit, tool
from agno.tools.function import ToolResult
from loguru import logger

from videodeepsearch.clients.storage.postgre import PostgresClient
from videodeepsearch.clients.storage.minio import MinioStorageClient


def extract_s3_minio_url(s3_link: str) -> tuple[str, str]:
    """Parse S3/MinIO URL to extract bucket and object name.

    Args:
        s3_link: S3 URL (e.g., "http://host/bucket/object" or "bucket/object")

    Returns:
        Tuple of (bucket_name, object_name)
    """
    parsed = urlparse(s3_link)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def format_duration(seconds: float) -> str:
    """Format seconds to HH:MM:SS format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# =============================================================================
# Response Models
# =============================================================================

class VideoInfo:
    """Simple video info container."""

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
            f"[{self.video_id[:8]}...] {self.filename} | "
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


class ProcessingStatus:
    """Processing status for a video."""

    def __init__(self, video_id: str):
        self.video_id = video_id
        self.stages: dict[str, bool] = {}

    def set_stage(self, stage: str, completed: bool) -> None:
        self.stages[stage] = completed

    def is_complete(self, stage: str) -> bool:
        return self.stages.get(stage, False)

    def completion_percentage(self) -> float:
        if not self.stages:
            return 0.0
        completed = sum(1 for v in self.stages.values() if v)
        return (completed / len(self.stages)) * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "stages": self.stages,
            "completion_percentage": self.completion_percentage(),
        }

    def repr(self) -> str:
        lines = [f"Processing Status for Video: {self.video_id}", "-" * 50]
        for stage, completed in sorted(self.stages.items()):
            status = "✅" if completed else "❌"
            lines.append(f"  {status} {stage}")
        lines.append(f"\nCompletion: {self.completion_percentage():.1f}%")
        return "\n".join(lines)


# =============================================================================
# Video Metadata Toolkit
# =============================================================================

class VideoMetadataToolkit(Toolkit):
    """Toolkit for video-level metadata and overview operations.

    Provides tools for:
    - Listing user videos with basic info
    - Getting detailed video metadata
    - Retrieving aggregated video summaries
    - Generating visual timelines
    - Checking processing pipeline status

    All tools return ToolResult for unified interface.
    """

    # Processing stages in order (for status checking)
    PROCESSING_STAGES = [
        "video_registration",
        "autoshot",
        "asr",
        "audio_segment",
        "image_extraction",
        "image_caption",
        "image_embedding",
        "image_ocr",
        "ocr_indexing",
        "segment_caption",
        "segment_embedding",
        "kg_graph",
        "arango_indexing",
    ]

    # Artifact types mapping for stage detection
    STAGE_ARTIFACT_TYPES = {
        "video_registration": "VideoArtifact",
        "autoshot": "AutoshotArtifact",
        "asr": "ASRArtifact",
        "audio_segment": "AudioSegmentArtifact",
        "image_extraction": "ImageArtifact",
        "image_caption": "ImageCaptionArtifact",
        "image_embedding": "ImageEmbeddingArtifact",
        "image_ocr": "ImageOCRArtifact",
        "ocr_indexing": "OCRElasticsearchArtifact",
        "segment_caption": "SegmentCaptionArtifact",
        "segment_embedding": "SegmentEmbeddingArtifact",
        "kg_graph": "KGGraphArtifact",
        "arango_indexing": "ArangoIndexingArtifact",
    }

    def __init__(
        self,
        postgres_client: PostgresClient,
        minio_client: MinioStorageClient,
    ):
        """Initialize the VideoMetadataToolkit.

        Args:
            postgres_client: PostgreSQL client for artifact metadata
            minio_client: MinIO storage client for fetching artifact data
        """
        self.postgres = postgres_client
        self.storage = minio_client
        super().__init__(name="Video Metadata Tools")

    async def _get_video_artifacts(
        self,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[VideoInfo]:
        """Fetch video artifacts from PostgreSQL.

        Args:
            user_id: Optional user ID filter
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of VideoInfo objects
        """
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
        """Get all artifact types that exist for a video.

        Args:
            video_id: Video ID to check

        Returns:
            Set of artifact type names
        """
        children = await self.postgres.get_children_artifact(video_id)
        return {child.artifact_type for child in children}

    # =========================================================================
    # List User Videos
    # =========================================================================

    @tool(
        description=(
            "List all videos for a user with basic metadata. "
            "Returns video IDs, filenames, duration, fps, and resolution. "
            "Use this to discover what videos are available for search."
        ),
        instructions=(
            "Use when: user wants to see available videos, "
            "need to find video IDs for further operations, "
            "starting a new search session."
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
        """List all videos for a specific user.

        Args:
            user_id: User ID to list videos for
            limit: Maximum number of videos to return (default 50)
            offset: Offset for pagination (default 0)

        Returns:
            ToolResult with list of videos and their metadata
        """
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

    # =========================================================================
    # Get Video Metadata
    # =========================================================================

    @tool(
        description=(
            "Get detailed metadata for a specific video. "
            "Returns comprehensive information including duration, fps, resolution, "
            "file location, and creation timestamp."
        ),
        instructions=(
            "Use when: need detailed info about a specific video, "
            "want to verify video properties before processing, "
            "checking video existence."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_video_metadata(
        self,
        video_id: str,
    ) -> ToolResult:
        """Get detailed metadata for a specific video.

        Args:
            video_id: Video ID to get metadata for

        Returns:
            ToolResult with detailed video metadata
        """
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
                f"MinIO URL: {artifact.minio_url}",
                "",
                "=== Full Metadata ===",
            ]

            for key, value in sorted(metadata.items()):
                lines.append(f"  {key}: {value}")

            return ToolResult(content="\n".join(lines))

        except Exception as e:
            logger.error(f"[VideoMetadataToolkit] get_video_metadata failed: {e}")
            return ToolResult(content=f"Error: Failed to get video metadata - {str(e)}")

    # =========================================================================
    # Get Video Summary
    # =========================================================================

    @tool(
        description=(
            "Get an aggregated summary of a video from all segment captions. "
            "Combines all segment-level captions into a coherent overview. "
            "Use this to understand what a video is about without watching it."
        ),
        instructions=(
            "Use when: user asks 'what is this video about', "
            "need a quick overview of video content, "
            "want to understand video theme before detailed search."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_video_summary(
        self,
        video_id: str,
        max_segments: int = 20,
    ) -> ToolResult:
        """Get an aggregated summary of a video from segment captions.

        Args:
            video_id: Video ID to summarize
            max_segments: Maximum number of segments to include (default 20)

        Returns:
            ToolResult with aggregated video summary
        """
        try:
            # Verify video exists
            video_artifact = await self.postgres.get_artifact(video_id)
            if video_artifact is None:
                return ToolResult(
                    content=f"Error: Video '{video_id}' not found."
                )

            # Get segment caption artifacts
            segment_artifacts = await self.postgres.get_children_artifact(
                artifact_id=video_id,
                filter_artifact_type=["SegmentCaptionArtifact"],
            )

            if not segment_artifacts:
                return ToolResult(
                    content=f"No segment captions found for video '{video_id}'. "
                    f"The video may still be processing or no captions were generated."
                )

            # Sort by created_at to maintain temporal order
            segment_artifacts.sort(key=lambda x: x.created_at)

            # Limit segments
            segment_artifacts = segment_artifacts[:max_segments]

            lines = [
                f"=== Video Summary for {video_id} ===",
                f"Total segments: {len(segment_artifacts)}",
                "",
            ]

            # Collect captions
            captions = []
            for i, artifact in enumerate(segment_artifacts):
                if artifact.minio_url:
                    bucket, object_name = extract_s3_minio_url(artifact.minio_url)
                    data = self.storage.read_json(bucket, object_name)

                    if data:
                        caption = data.get("caption", "")
                        summary = data.get("summary_caption", "")
                        audio_text = data.get("audio_text", "")
                        start_time = data.get("start_timestamp", "00:00:00.000")
                        end_time = data.get("end_timestamp", "00:00:00.000")

                        segment_text = caption or summary or audio_text

                        if segment_text:
                            captions.append({
                                "index": i + 1,
                                "start": start_time,
                                "end": end_time,
                                "text": segment_text[:500],  # Limit text length
                            })

                            lines.append(
                                f"[{i + 1}] {start_time} - {end_time}"
                            )
                            lines.append(f"    {segment_text[:200]}...")
                            lines.append("")

            if not captions:
                return ToolResult(
                    content=f"No caption content could be loaded for video '{video_id}'."
                )

            # Add summary statistics
            lines.append("---")
            lines.append(f"Segments shown: {len(captions)}")

            return ToolResult(content="\n".join(lines))

        except Exception as e:
            logger.error(f"[VideoMetadataToolkit] get_video_summary failed: {e}")
            return ToolResult(content=f"Error: Failed to get video summary - {str(e)}")

    # =========================================================================
    # Get Video Timeline
    # =========================================================================

    @tool(
        description=(
            "Get a visual timeline of video segments with timestamps and captions. "
            "Shows the temporal structure of the video with segment boundaries. "
            "Use this to navigate through video content chronologically."
        ),
        instructions=(
            "Use when: user wants to see video structure, "
            "need to find specific time ranges, "
            "planning temporal navigation."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_video_timeline(
        self,
        video_id: str,
        granularity: Literal["segment", "shot", "minute"] = "segment",
    ) -> ToolResult:
        """Get a visual timeline of video segments.

        Args:
            video_id: Video ID to generate timeline for
            granularity: Timeline granularity ('segment', 'shot', or 'minute')

        Returns:
            ToolResult with visual timeline
        """
        try:
            # Verify video exists
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
                # Get segment caption artifacts
                segment_artifacts = await self.postgres.get_children_artifact(
                    artifact_id=video_id,
                    filter_artifact_type=["SegmentCaptionArtifact"],
                )

                if not segment_artifacts:
                    lines.append("No segments found. Video may still be processing.")
                    return ToolResult(content="\n".join(lines))

                # Sort by start time
                segments_data = []
                for artifact in segment_artifacts:
                    if artifact.minio_url:
                        bucket, object_name = extract_s3_minio_url(artifact.minio_url)
                        data = self.storage.read_json(bucket, object_name)

                        if data:
                            segments_data.append({
                                "start_frame": data.get("start_frame", 0),
                                "end_frame": data.get("end_frame", 0),
                                "start_time": data.get("start_timestamp", "00:00:00.000"),
                                "end_time": data.get("end_timestamp", "00:00:00.000"),
                                "caption": data.get("caption", "")[:100],
                            })

                segments_data.sort(key=lambda x: x["start_frame"])

                # Build timeline
                lines.append(f"{'Time Range':<25} | {'Frames':<15} | Caption")
                lines.append("-" * 80)

                for seg in segments_data:
                    time_range = f"{seg['start_time']} - {seg['end_time']}"
                    frame_range = f"{seg['start_frame']} - {seg['end_frame']}"
                    caption = seg['caption'][:40] + "..." if len(seg['caption']) > 40 else seg['caption']

                    lines.append(f"{time_range:<25} | {frame_range:<15} | {caption}")

                lines.append("")
                lines.append(f"Total segments: {len(segments_data)}")

            elif granularity == "shot":
                # Get autoshot artifacts
                autoshot_artifacts = await self.postgres.get_children_artifact(
                    artifact_id=video_id,
                    filter_artifact_type=["AutoshotArtifact"],
                )

                if not autoshot_artifacts:
                    lines.append("No shot boundaries found. Video may still be processing.")
                    return ToolResult(content="\n".join(lines))

                # Load shot data
                for artifact in autoshot_artifacts:
                    if artifact.minio_url:
                        bucket, object_name = extract_s3_minio_url(artifact.minio_url)
                        data = self.storage.read_json(bucket, object_name)

                        if data:
                            shots = data.get("shots", [])
                            lines.append(f"Total shots detected: {len(shots)}")
                            lines.append("")
                            lines.append(f"{'Shot #':<8} | {'Start Frame':<12} | {'End Frame':<12}")
                            lines.append("-" * 40)

                            for i, shot in enumerate(shots[:50]):  # Limit to 50 shots
                                lines.append(
                                    f"{i + 1:<8} | {shot.get('start_frame', 0):<12} | "
                                    f"{shot.get('end_frame', 0):<12}"
                                )

                            if len(shots) > 50:
                                lines.append(f"... and {len(shots) - 50} more shots")

                            break

            elif granularity == "minute":
                # Get segments and group by minute
                segment_artifacts = await self.postgres.get_children_artifact(
                    artifact_id=video_id,
                    filter_artifact_type=["SegmentCaptionArtifact"],
                )

                if not segment_artifacts:
                    lines.append("No segments found. Video may still be processing.")
                    return ToolResult(content="\n".join(lines))

                # Collect segments
                segments_data = []
                for artifact in segment_artifacts:
                    if artifact.minio_url:
                        bucket, object_name = extract_s3_minio_url(artifact.minio_url)
                        data = self.storage.read_json(bucket, object_name)

                        if data:
                            start_sec = data.get("start_sec", 0)
                            segments_data.append({
                                "start_sec": start_sec,
                                "minute": int(start_sec // 60),
                                "caption": data.get("caption", "")[:100],
                            })

                # Group by minute
                from collections import defaultdict
                minute_groups = defaultdict(list)
                for seg in segments_data:
                    minute_groups[seg["minute"]].append(seg)

                lines.append(f"{'Minute':<8} | {'Segments':<10} | Sample Caption")
                lines.append("-" * 60)

                for minute in sorted(minute_groups.keys()):
                    segments = minute_groups[minute]
                    sample = segments[0]["caption"][:40] + "..." if segments[0]["caption"] else "N/A"
                    lines.append(f"{minute:<8} | {len(segments):<10} | {sample}")

                lines.append("")
                lines.append(f"Total minutes with content: {len(minute_groups)}")

            return ToolResult(content="\n".join(lines))

        except Exception as e:
            logger.error(f"[VideoMetadataToolkit] get_video_timeline failed: {e}")
            return ToolResult(content=f"Error: Failed to get video timeline - {str(e)}")

    # =========================================================================
    # Get Processing Status
    # =========================================================================

    @tool(
        description=(
            "Check which processing stages are complete for a video. "
            "Shows the status of each pipeline stage: autoshot, ASR, captioning, "
            "embedding, OCR, knowledge graph, etc."
        ),
        instructions=(
            "Use when: video was recently uploaded and may still be processing, "
            "search returns no results, "
            "verifying that specific processing stages completed."
        ),
        cache_results=False,
    )
    async def get_processing_status(
        self,
        video_id: str,
    ) -> ToolResult:
        """Check processing pipeline status for a video.

        Args:
            video_id: Video ID to check status for

        Returns:
            ToolResult with processing status for each stage
        """
        try:
            # Verify video exists
            video_artifact = await self.postgres.get_artifact(video_id)
            if video_artifact is None:
                return ToolResult(
                    content=f"Error: Video '{video_id}' not found."
                )

            # Get all children artifact types
            artifact_types = await self._get_all_children_artifact_types(video_id)

            # Build status
            status = ProcessingStatus(video_id)

            for stage, artifact_type in self.STAGE_ARTIFACT_TYPES.items():
                completed = artifact_type in artifact_types
                status.set_stage(stage, completed)

            return ToolResult(content=status.repr())

        except Exception as e:
            logger.error(f"[VideoMetadataToolkit] get_processing_status failed: {e}")
            return ToolResult(content=f"Error: Failed to get processing status - {str(e)}")

    # =========================================================================
    # List All Videos (Admin)
    # =========================================================================

    @tool(
        description=(
            "List all videos in the system (admin operation). "
            "Returns all videos regardless of user. "
            "Use with caution in production environments."
        ),
        instructions=(
            "Use when: need to see all videos across users, "
            "admin/debug operations."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def list_all_videos(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> ToolResult:
        """List all videos in the system.

        Args:
            limit: Maximum number of videos to return (default 50)
            offset: Offset for pagination (default 0)

        Returns:
            ToolResult with list of all videos
        """
        try:
            videos = await self._get_video_artifacts(
                user_id=None,
                limit=limit,
                offset=offset,
            )

            if not videos:
                return ToolResult(content="No videos found in the system.")

            lines = [
                f"Found {len(videos)} video(s) in the system:",
                "",
            ]

            for i, video in enumerate(videos):
                lines.append(f"{i + 1}. {video.brief_repr()} (user: {video.user_id[:8]}...)")

            lines.append("")
            lines.append(f"Showing {len(videos)} results (offset: {offset}, limit: {limit})")

            return ToolResult(content="\n".join(lines))

        except Exception as e:
            logger.error(f"[VideoMetadataToolkit] list_all_videos failed: {e}")
            return ToolResult(content=f"Error: Failed to list videos - {str(e)}")

    # =========================================================================
    # Get Video Statistics
    # =========================================================================

    @tool(
        description=(
            "Get statistics about a video's processed content. "
            "Returns counts of images, segments, OCR results, and other artifacts. "
            "Use this to understand the depth of processing for a video."
        ),
        instructions=(
            "Use when: need to know how much content was extracted, "
            "comparing processing results across videos."
        ),
        cache_results=True,
        cache_ttl=1800,
    )
    async def get_video_statistics(
        self,
        video_id: str,
    ) -> ToolResult:
        """Get statistics about a video's processed content.

        Args:
            video_id: Video ID to get statistics for

        Returns:
            ToolResult with content statistics
        """
        try:
            # Verify video exists
            video_artifact = await self.postgres.get_artifact(video_id)
            if video_artifact is None:
                return ToolResult(
                    content=f"Error: Video '{video_id}' not found."
                )

            # Get all children
            children = await self.postgres.get_children_artifact(video_id)

            # Count by type
            from collections import Counter
            type_counts = Counter(child.artifact_type for child in children)

            lines = [
                f"=== Video Statistics for {video_id} ===",
                "",
            ]

            # Group by category
            categories = {
                "Video Processing": ["AutoshotArtifact"],
                "Audio": ["ASRArtifact", "AudioSegmentArtifact"],
                "Images": ["ImageArtifact", "ImageCaptionArtifact", "ImageEmbeddingArtifact", "ImageOCRArtifact"],
                "Segments": ["SegmentCaptionArtifact", "SegmentEmbeddingArtifact"],
                "Knowledge Graph": ["KGGraphArtifact", "ArangoIndexingArtifact"],
                "Indexing": ["OCRElasticsearchArtifact"],
            }

            for category, types in categories.items():
                total = sum(type_counts.get(t, 0) for t in types)
                lines.append(f"📁 {category}: {total} artifact(s)")

                for artifact_type in types:
                    count = type_counts.get(artifact_type, 0)
                    if count > 0:
                        # Clean up name
                        name = artifact_type.replace("Artifact", "")
                        lines.append(f"   └─ {name}: {count}")

                lines.append("")

            # Total
            total_artifacts = len(children)
            lines.append(f"Total Artifacts: {total_artifacts}")

            return ToolResult(content="\n".join(lines))

        except Exception as e:
            logger.error(f"[VideoMetadataToolkit] get_video_statistics failed: {e}")
            return ToolResult(content=f"Error: Failed to get video statistics - {str(e)}")


__all__ = ["VideoMetadataToolkit"]