"""Cache key functions for Prefect task caching.

These functions generate deterministic cache keys for tasks,
ignoring auto-generated fields like artifact_id and temp file paths.
"""

import hashlib
from typing import Any


def _hash_string(s: str, length: int = 12) -> str:
    """Hash a string to a fixed-length hex digest.

    Args:
        s: String to hash
        length: Length of the output hash (default 12 characters)

    Returns:
        Hex digest of the hash
    """
    return hashlib.md5(s.encode()).hexdigest()[:length]


def video_registration_cache_key(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    """Cache key for video_registration task that takes VideoInput.

    Uses video_id + video_s3_url to identify the same video content.

    Usage in tasks.yaml:
        cache_key_fn: "video_registration_cache_key"

    Args:
        _context: Prefect task run context (unused)
        parameters: Task parameters containing 'video_input'

    Returns:
        Cache key string or None if video_input not found
    """
    video_input = parameters.get("video_input")
    if video_input is None:
        return None

    # Hash the URL to avoid MinIO-invalid characters
    url_hash = _hash_string(video_input.video_s3_url)
    return f"video-reg-{video_input.video_id}-{url_hash}"


def video_artifact_cache_key(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    """Cache key for tasks that take VideoArtifact input.

    Uses video_id + video_minio_url to identify the same video content,
    ignoring auto-generated artifact_id.

    Usage in tasks.yaml:
        cache_key_fn: "video_artifact_cache_key"

    Args:
        _context: Prefect task run context (unused)
        parameters: Task parameters containing 'video_artifact'

    Returns:
        Cache key string or None if video_artifact not found
    """
    video_artifact = parameters.get("video_artifact")
    if video_artifact is None:
        return None

    # Hash the URL to avoid MinIO-invalid characters
    url_hash = _hash_string(video_artifact.video_minio_url)
    return f"video-{video_artifact.video_id}-{url_hash}"


def autoshot_artifact_cache_key(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    """Cache key for tasks that take AutoshotArtifact input.

    Uses video_id + video_minio_url to identify the same video content,
    ignoring auto-generated artifact_id.

    Usage in tasks.yaml:
        cache_key_fn: "autoshot_artifact_cache_key"

    Args:
        _context: Prefect task run context (unused)
        parameters: Task parameters containing 'autoshot_artifact'

    Returns:
        Cache key string or None if autoshot_artifact not found
    """
    autoshot_artifact = parameters.get("autoshot_artifact")
    if autoshot_artifact is None:
        return None

    # Hash the URL to avoid MinIO-invalid characters
    url_hash = _hash_string(autoshot_artifact.related_video_minio_url)
    return f"autoshot-{autoshot_artifact.related_video_id}-{url_hash}"


def asr_batch_cache_key(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    items = parameters.get("items")
    if not items:
        return None

    key_parts = []
    video_id = None
    for item in items:
        autoshot_artifact, start_frame, end_frame, _ = item
        if video_id is None:
            video_id = autoshot_artifact.related_video_id
        key_parts.append(f"{start_frame}-{end_frame}")

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)

    return f"asr-{video_id}-{key_hash}"


def audio_segment_cache_key(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    """Cache key for audio_segment task that takes list[ASRArtifact].

    Uses video_id + sorted frame numbers to create a deterministic cache key,
    ignoring auto-generated artifact_id.

    Usage in tasks.yaml:
        cache_key_fn: "audio_segment_cache_key"

    Args:
        _context: Prefect task run context (unused)
        parameters: Task parameters containing 'asr_artifacts' as list of ASRArtifact

    Returns:
        Cache key string or None if asr_artifacts not found
    """
    asr_artifacts = parameters.get("asr_artifacts")
    if not asr_artifacts:
        return None

    video_id = None
    key_parts = []

    for artifact in asr_artifacts:
        if video_id is None:
            video_id = artifact.related_video_id
        frame_num = artifact.metadata.get("frame_num", [0, 0]) if artifact.metadata else [0, 0]
        start_frame, end_frame = frame_num[0], frame_num[1]
        key_parts.append(f"{start_frame}-{end_frame}")

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)

    return f"audio-seg-{video_id}-{key_hash}"


def segment_embedding_cache_key(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    """Cache key for segment_embedding_chunk_task that takes list[AudioSegmentArtifact].

    Uses video_id + sorted frame ranges to create a deterministic cache key,
    ignoring auto-generated artifact_id.

    Usage in tasks.yaml:
        cache_key_fn: "segment_embedding_cache_key"

    Args:
        _context: Prefect task run context (unused)
        parameters: Task parameters containing 'segments' as list of AudioSegmentArtifact

    Returns:
        Cache key string or None if segments not found
    """
    segments = parameters.get("segments")
    if not segments:
        return None

    video_id = None
    key_parts = []

    for artifact in segments:
        if video_id is None:
            video_id = artifact.related_video_id
        key_parts.append(f"{artifact.start_frame}-{artifact.end_frame}")

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)

    return f"seg-emb-{video_id}-{key_hash}"


def segment_caption_cache_key(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    """Cache key for segment_caption_chunk_task that takes list[AudioSegmentArtifact].

    Uses video_id + sorted frame ranges to create a deterministic cache key.

    Usage in tasks.yaml:
        cache_key_fn: "segment_caption_cache_key"

    Args:
        _context: Prefect task run context (unused)
        parameters: Task parameters containing 'segments' as list of AudioSegmentArtifact

    Returns:
        Cache key string or None if segments not found
    """
    segments = parameters.get("segments")
    if not segments:
        return None

    video_id = None
    key_parts = []

    for artifact in segments:
        if video_id is None:
            video_id = artifact.related_video_id
        key_parts.append(f"{artifact.start_frame}-{artifact.end_frame}")

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)

    return f"seg-cap-{video_id}-{key_hash}"


def segment_caption_embedding_cache_key(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    """Cache key for segment_caption_embedding_chunk_task that takes list[SegmentCaptionArtifact].

    Uses video_id + sorted frame ranges to create a deterministic cache key.

    Usage in tasks.yaml:
        cache_key_fn: "segment_caption_embedding_cache_key"

    Args:
        _context: Prefect task run context (unused)
        parameters: Task parameters containing 'items' as list of SegmentCaptionArtifact

    Returns:
        Cache key string or None if items not found
    """
    items = parameters.get("items")
    if not items:
        return None

    video_id = None
    key_parts = []

    for artifact in items:
        if video_id is None:
            video_id = artifact.related_video_id
        key_parts.append(f"{artifact.start_frame}-{artifact.end_frame}")

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)

    return f"seg-cap-emb-{video_id}-{key_hash}"


def segment_caption_multimodal_embedding_cache_key(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    """Cache key for segment_caption_multimodal_embedding_chunk_task that takes list[SegmentCaptionArtifact].

    Uses video_id + sorted frame ranges to create a deterministic cache key.

    Usage in tasks.yaml:
        cache_key_fn: "segment_caption_multimodal_embedding_cache_key"

    Args:
        _context: Prefect task run context (unused)
        parameters: Task parameters containing 'items' as list of SegmentCaptionArtifact

    Returns:
        Cache key string or None if items not found
    """
    items = parameters.get("items")
    if not items:
        return None

    video_id = None
    key_parts = []

    for artifact in items:
        if video_id is None:
            video_id = artifact.related_video_id
        key_parts.append(f"{artifact.start_frame}-{artifact.end_frame}")

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)

    return f"seg-cap-mm-emb-{video_id}-{key_hash}"


def image_batch_cache_key_caption(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    items = parameters.get("items")
    if not items:
        return None

    key_parts = []
    video_id = None
    for artifact in items:
        if video_id is None:
            video_id = artifact.related_video_id
        if artifact.object_name:
            key_parts.append(artifact.object_name)
        else:
            key_parts.append(str(artifact.frame_index))

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)

    return f"img-{video_id}-{key_hash}"


def image_batch_cache_key_embedding(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    items = parameters.get("items")
    if not items:
        return None

    key_parts = []
    video_id = None
    for artifact in items:
        if video_id is None:
            video_id = artifact.related_video_id
        if artifact.object_name:
            key_parts.append(artifact.object_name)
        else:
            key_parts.append(str(artifact.frame_index))

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)

    return f"img-emb-{video_id}-{key_hash}"


def image_batch_cache_key_ocr(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    items = parameters.get("items")
    if not items:
        return None

    key_parts = []
    video_id = None
    for artifact in items:
        if video_id is None:
            video_id = artifact.related_video_id
        if artifact.object_name:
            key_parts.append(artifact.object_name)
        else:
            key_parts.append(str(artifact.frame_index))

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)

    return f"img-ocr-{video_id}-{key_hash}"


def image_extraction_batch_cache_key(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    """Cache key for image_extraction task that takes list[(AutoshotArtifact, frame_index)].

    Uses video_id + video_url + sorted frame indices to create a deterministic cache key,
    ignoring auto-generated artifact_id.

    Usage in tasks.yaml:
        cache_key_fn: "image_extraction_batch_cache_key"

    Args:
        _context: Prefect task run context (unused)
        parameters: Task parameters containing 'items' as list of (AutoshotArtifact, int) tuples

    Returns:
        Cache key string or None if items not found
    """
    items = parameters.get("items")
    if not items:
        return None

    video_id = None
    video_url = None
    frame_indices = []

    for autoshot_artifact, frame_index in items:
        if video_id is None:
            video_id = autoshot_artifact.related_video_id
            video_url = autoshot_artifact.related_video_minio_url
        frame_indices.append(frame_index)

    frame_indices.sort()
    frames_string = ",".join(str(idx) for idx in frame_indices)
    frames_hash = _hash_string(frames_string)

    # Hash the URL to avoid MinIO-invalid characters
    url_hash = _hash_string(video_url) if video_url else "no-url"

    return f"img-extract-{video_id}-{url_hash}-{frames_hash}"


def caption_embedding_batch_cache_key(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    """Cache key for image_caption_embedding task that takes list[ImageCaptionArtifact].

    Uses video_id + sorted frame indices to create a deterministic cache key.

    Usage in tasks.yaml:
        cache_key_fn: "caption_embedding_batch_cache_key"

    Args:
        _context: Prefect task run context (unused)
        parameters: Task parameters containing 'items' as list of ImageCaptionArtifact

    Returns:
        Cache key string or None if items not found
    """
    items = parameters.get("items")
    if not items:
        return None

    video_id = None
    key_parts = []

    for artifact in items:
        if video_id is None:
            video_id = artifact.related_video_id
        key_parts.append(str(artifact.frame_index))

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)

    return f"cap-emb-{video_id}-{key_hash}"


def caption_multimodal_embedding_batch_cache_key(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    """Cache key for image_caption_multimodal_embedding task that takes list[ImageCaptionArtifact].

    Uses video_id + sorted frame indices to create a deterministic cache key.

    Usage in tasks.yaml:
        cache_key_fn: "caption_multimodal_embedding_batch_cache_key"

    Args:
        _context: Prefect task run context (unused)
        parameters: Task parameters containing 'items' as list of ImageCaptionArtifact

    Returns:
        Cache key string or None if items not found
    """
    items = parameters.get("items")
    if not items:
        return None

    video_id = None
    key_parts = []

    for artifact in items:
        if video_id is None:
            video_id = artifact.related_video_id
        key_parts.append(str(artifact.frame_index))

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)

    return f"cap-mm-emb-{video_id}-{key_hash}"


def image_qdrant_indexing_cache_key(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    """Cache key for image_qdrant_indexing_chunk_task that takes list[ImageEmbeddingArtifact].

    Uses video_id + sorted frame indices to create a deterministic cache key,
    ignoring auto-generated artifact_id.

    Usage in tasks.yaml:
        cache_key_fn: "image_qdrant_indexing_cache_key"

    Args:
        _context: Prefect task run context (unused)
        parameters: Task parameters containing 'items' as list of ImageEmbeddingArtifact

    Returns:
        Cache key string or None if items not found
    """
    items = parameters.get("items")
    if not items:
        return None

    video_id = None
    key_parts = []

    for artifact in items:
        if video_id is None:
            video_id = artifact.related_video_id
        key_parts.append(str(artifact.frame_index))

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)

    return f"img-qdrant-{video_id}-{key_hash}"


def caption_qdrant_indexing_cache_key(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    """Cache key for caption_qdrant_indexing_chunk_task that takes text_items and mm_items.

    Uses video_id + sorted frame indices to create a deterministic cache key,
    ignoring auto-generated artifact_id.

    Usage in tasks.yaml:
        cache_key_fn: "caption_qdrant_indexing_cache_key"

    Args:
        _context: Prefect task run context (unused)
        parameters: Task parameters containing 'text_items' and 'mm_items' as aligned lists

    Returns:
        Cache key string or None if items not found
    """
    text_items = parameters.get("text_items")
    if not text_items:
        return None

    video_id = None
    key_parts = []

    for artifact in text_items:
        if video_id is None:
            video_id = artifact.related_video_id
        key_parts.append(str(artifact.frame_index))

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)

    return f"cap-qdrant-{video_id}-{key_hash}"


def segment_qdrant_indexing_cache_key(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    """Cache key for segment_qdrant_indexing_chunk_task that takes list[SegmentEmbeddingArtifact].

    Uses video_id + sorted frame ranges to create a deterministic cache key,
    ignoring auto-generated artifact_id.

    Usage in tasks.yaml:
        cache_key_fn: "segment_qdrant_indexing_cache_key"

    Args:
        _context: Prefect task run context (unused)
        parameters: Task parameters containing 'items' as list of SegmentEmbeddingArtifact

    Returns:
        Cache key string or None if items not found
    """
    items = parameters.get("items")
    if not items:
        return None

    video_id = None
    key_parts = []

    for artifact in items:
        if video_id is None:
            video_id = artifact.related_video_id
        key_parts.append(f"{artifact.start_frame}-{artifact.end_frame}")

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)

    return f"seg-qdrant-{video_id}-{key_hash}"


def segment_caption_qdrant_indexing_cache_key(
    _context: Any,
    parameters: dict[str, Any],
) -> str | None:
    """Cache key for segment_caption_qdrant_indexing_chunk_task that takes text_items and mm_items.

    Uses video_id + sorted frame ranges to create a deterministic cache key,
    ignoring auto-generated artifact_id.

    Usage in tasks.yaml:
        cache_key_fn: "segment_caption_qdrant_indexing_cache_key"

    Args:
        _context: Prefect task run context (unused)
        parameters: Task parameters containing 'text_items' and 'mm_items' as aligned lists

    Returns:
        Cache key string or None if items not found
    """
    text_items = parameters.get("text_items")
    if not text_items:
        return None

    video_id = None
    key_parts = []

    for artifact in text_items:
        if video_id is None:
            video_id = artifact.related_video_id
        key_parts.append(f"{artifact.start_frame}-{artifact.end_frame}")

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)

    return f"seg-cap-qdrant-{video_id}-{key_hash}"


CACHE_KEY_FUNCTIONS: dict[str, Any] = {
    "video_registration_cache_key": video_registration_cache_key,
    "video_artifact_cache_key": video_artifact_cache_key,
    "autoshot_artifact_cache_key": autoshot_artifact_cache_key,
    "asr_batch_cache_key": asr_batch_cache_key,
    "audio_segment_cache_key": audio_segment_cache_key,
    "segment_embedding_cache_key": segment_embedding_cache_key,
    "segment_caption_cache_key": segment_caption_cache_key,
    "segment_caption_embedding_cache_key": segment_caption_embedding_cache_key,
    "segment_caption_multimodal_embedding_cache_key": segment_caption_multimodal_embedding_cache_key,
    "image_batch_cache_key_caption": image_batch_cache_key_caption,
    "image_batch_cache_key_embedding": image_batch_cache_key_embedding,
    "image_batch_cache_key_ocr": image_batch_cache_key_ocr,
    "image_extraction_batch_cache_key": image_extraction_batch_cache_key,
    "caption_embedding_batch_cache_key": caption_embedding_batch_cache_key,
    "caption_multimodal_embedding_batch_cache_key": caption_multimodal_embedding_batch_cache_key,
    "image_qdrant_indexing_cache_key": image_qdrant_indexing_cache_key,
    "caption_qdrant_indexing_cache_key": caption_qdrant_indexing_cache_key,
    "segment_qdrant_indexing_cache_key": segment_qdrant_indexing_cache_key,
    "segment_caption_qdrant_indexing_cache_key": segment_caption_qdrant_indexing_cache_key,
}
