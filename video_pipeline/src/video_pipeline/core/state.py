from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.client.storage.minio import MinioStorageClient
    from core.storage.pg_tracker import ArtifactPersistentVisitor



class AppState:
    _instance: AppState | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    

    minio_client: MinioStorageClient = None #type:ignore
    artifact_visitor: ArtifactPersistentVisitor = None #type:ignore
