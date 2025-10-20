from functools import lru_cache
from fastapi import Request
from core.pipeline.tracker import ArtifactTracker
from core.storage import StorageClient
from core.management.cleanup import ArtifactDeleter
from core.management.status import VideoStatusManager

@lru_cache(maxsize=1)
def get_artifact_tracker(request: Request) -> ArtifactTracker:
    return request.app.state.artifact_tracker

@lru_cache(maxsize=1)
def get_storage_client(request: Request) -> StorageClient:
    return request.app.state.storage_client



@lru_cache(maxsize=1)
def get_artifact_deleter(request: Request) -> ArtifactDeleter:
    return request.app.state.artifact_deleter



@lru_cache(maxsize=1)
def get_video_status_manager(request: Request) -> VideoStatusManager:
    return request.app.state.video_status