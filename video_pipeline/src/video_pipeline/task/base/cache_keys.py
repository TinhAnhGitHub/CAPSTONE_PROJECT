"""Cache key functions for Prefect task caching."""

import hashlib
from typing import Any


def _hash_string(s: str, length: int = 12) -> str:
    """Hash string to fixed-length hex digest."""
    return hashlib.md5(s.encode()).hexdigest()[:length]


def video_registration_cache_key(
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + url hash."""
    video_input = parameters.get("video_input")
    if video_input is None:
        return None
    url_hash = _hash_string(video_input.video_s3_url)
    print(f"{url_hash=}")
    return f"video-reg-{video_input.video_id}-{url_hash}"


def video_artifact_cache_key(
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + url hash."""
    video_artifact = parameters.get("video_artifact")
    if video_artifact is None:
        return None
    url_hash = _hash_string(video_artifact.video_minio_url)
    return f"video-{video_artifact.video_id}-{url_hash}"


def autoshot_artifact_cache_key(
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key for autoshot task - uses video_artifact parameter."""
    video_artifact = parameters.get("video_artifact")
    if video_artifact is None:
        return None
    url_hash = _hash_string(video_artifact.video_minio_url)
    return f"autoshot-{video_artifact.video_id}-{url_hash}"


def asr_batch_cache_key(
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + sorted frame ranges."""
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
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + sorted frame ranges."""
    asr_artifacts = parameters.get("asr_artifacts")
    if not asr_artifacts:
        return None

    video_id = None
    key_parts = []
    for artifact in asr_artifacts:
        if video_id is None:
            video_id = artifact.related_video_id
        frame_num = artifact.metadata.get("frame_num", [0, 0]) if artifact.metadata else [0, 0]
        key_parts.append(f"{frame_num[0]}-{frame_num[1]}")

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)
    return f"audio-seg-{video_id}-{key_hash}"


def segment_embedding_cache_key(
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + sorted frame ranges."""
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
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + sorted frame ranges."""
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
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + sorted frame ranges."""
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


def image_batch_cache_key_caption(
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + sorted object names/frame indices."""
    items = parameters.get("items")
    if not items:
        return None

    key_parts = []
    video_id = None
    for artifact in items:
        if video_id is None:
            video_id = artifact.related_video_id
        key_parts.append(artifact.object_name or str(artifact.frame_index))

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)
    return f"img-{video_id}-{key_hash}"


def image_batch_cache_key_embedding(
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + sorted object names/frame indices."""
    items = parameters.get("items")
    if not items:
        return None

    key_parts = []
    video_id = None
    for artifact in items:
        if video_id is None:
            video_id = artifact.related_video_id
        key_parts.append(artifact.object_name or str(artifact.frame_index))

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)
    return f"img-emb-{video_id}-{key_hash}"


def image_batch_cache_key_ocr(
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + sorted object names/frame indices."""
    items = parameters.get("items")
    if not items:
        return None

    key_parts = []
    video_id = None
    for artifact in items:
        if video_id is None:
            video_id = artifact.related_video_id
        key_parts.append(artifact.object_name or str(artifact.frame_index))

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)
    return f"img-ocr-{video_id}-{key_hash}"


def image_extraction_batch_cache_key(
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + url hash + sorted frame indices."""
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
    url_hash = _hash_string(video_url) if video_url else "no-url"

    return f"img-extract-{video_id}-{url_hash}-{frames_hash}"


def caption_embedding_batch_cache_key(
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + sorted frame indices."""
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


def image_qdrant_indexing_cache_key(
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + sorted frame indices."""
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


def segment_qdrant_indexing_cache_key(
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + sorted frame ranges."""
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


def kg_pipeline_cache_key(
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + sorted frame ranges + text hash."""
    segments = parameters.get("segments")
    if not segments:
        return None

    video_id = None
    frame_parts = []
    text_parts = []

    for artifact in segments:
        if video_id is None:
            video_id = artifact.related_video_id
        frame_parts.append(f"{artifact.start_frame}-{artifact.end_frame}")
        text_parts.append(f"{artifact.audio_text}|{artifact.summary_caption}")

    frame_string = "|".join(sorted(frame_parts))
    text_string = "|".join(sorted(text_parts))

    frame_hash = _hash_string(frame_string)
    text_hash = _hash_string(text_string)

    return f"kg-pipeline-{video_id}-{frame_hash}-{text_hash}"


def audio_transcript_embedding_cache_key(
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + sorted frame ranges + audio_text hash."""
    items = parameters.get("items")
    if not items:
        return None

    video_id = None
    key_parts = []

    for artifact in items:
        if video_id is None:
            video_id = artifact.related_video_id
        key_parts.append(f"{artifact.start_frame}-{artifact.end_frame}")
        if artifact.audio_text:
            key_parts.append(_hash_string(artifact.audio_text, 8))

    key_string = "|".join(sorted(key_parts))
    key_hash = _hash_string(key_string)
    return f"audio-trans-emb-{video_id}-{key_hash}"


def audio_transcript_qdrant_indexing_cache_key(
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + sorted frame ranges."""
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
    return f"audio-trans-qdrant-{video_id}-{key_hash}"


def image_caption_qdrant_indexing_cache_key(
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + sorted frame indices."""
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
    return f"img-cap-qdrant-{video_id}-{key_hash}"


def segment_caption_qdrant_indexing_cache_key(
    _context: Any, parameters: dict[str, Any]
) -> str | None:
    """Cache key: video_id + sorted frame ranges."""
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
    "image_batch_cache_key_caption": image_batch_cache_key_caption,
    "image_batch_cache_key_embedding": image_batch_cache_key_embedding,
    "image_batch_cache_key_ocr": image_batch_cache_key_ocr,
    "image_extraction_batch_cache_key": image_extraction_batch_cache_key,
    "caption_embedding_batch_cache_key": caption_embedding_batch_cache_key,
    "image_qdrant_indexing_cache_key": image_qdrant_indexing_cache_key,
    "segment_qdrant_indexing_cache_key": segment_qdrant_indexing_cache_key,
    "kg_pipeline_cache_key": kg_pipeline_cache_key,
    "audio_transcript_embedding_cache_key": audio_transcript_embedding_cache_key,
    "audio_transcript_qdrant_indexing_cache_key": audio_transcript_qdrant_indexing_cache_key,
    "image_caption_qdrant_indexing_cache_key": image_caption_qdrant_indexing_cache_key,
    "segment_caption_qdrant_indexing_cache_key": segment_caption_qdrant_indexing_cache_key,
}