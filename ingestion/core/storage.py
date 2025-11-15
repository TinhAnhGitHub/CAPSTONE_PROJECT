from __future__ import annotations
import os
import json
from datetime import timedelta
from io import BytesIO
from typing import Any, BinaryIO, Dict, Iterable, Optional
import socket

import urllib3
from urllib3.util import Timeout

from loguru import logger
from minio import Minio
from minio.error import S3Error

from core.config.storage import MinioSettings


class StorageError(RuntimeError):
    """Raised when MinIO storage operations fail."""
    pass

class StorageClient:
    def __init__(self, settings: MinioSettings ) -> None:
        self.settings = settings 
        timeout = Timeout(connect=5.0, read=20.0)  
        self._http_client = urllib3.PoolManager(
            maxsize=200,
            timeout=timeout,
            block=False,
        )
        self.client = Minio(
            endpoint=f"{settings.host}:{settings.port}",
            access_key=settings.access_key,
            secret_key=settings.secret_key,
            secure=settings.secure,
            http_client=self._http_client,
        )

    def _ensure_bucket(self, bucket: str) -> None:
        
        try:
            exists = self.client.bucket_exists(bucket)
            if not exists:
                logger.info(f"Bucket named: {bucket} does not exist, creating")
                self.client.make_bucket(bucket)
    
        except S3Error as exc:
            logger.error("MinIO bucket check failed for %s: %s", bucket, exc)
            raise StorageError(f"Failed to ensure bucket {bucket}: {exc}") from exc

    def upload_fileobj(
        self,
        bucket: str,
        object_name: str,
        file_obj: BinaryIO,
        *,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Upload a file-like object to the specified bucket and return an s3 URI."""

        self._ensure_bucket(bucket)
        length = -1
        try:
            current_pos = file_obj.tell()
            file_obj.seek(0, os.SEEK_END)
            end_pos = file_obj.tell()
            length = end_pos - current_pos
            file_obj.seek(current_pos)
        except:
            length = -1
        
        put_kwargs: dict[str, Any] = dict(
            bucket_name=bucket,
            object_name=object_name,
            data=file_obj,
            length=length,
            content_type=content_type,
            metadata=metadata,  # type: ignore
        )
        try:
            self.client.put_object(**put_kwargs)
            uri = f"s3://{bucket}/{object_name}"
            logger.info(f"Uploaded object {uri}")
            return uri
        except S3Error as exc:
            logger.exception("Failed to upload %s to bucket %s", object_name, bucket)
            raise StorageError(f"Upload failed for {object_name}: {exc}") from exc

    def put_json(
        self,
        bucket: str,
        object_name: str,
        payload: Dict[str, Any],
        *,
        content_type: str = "application/json",
    ) -> str:
        buffer = BytesIO(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        return self.upload_fileobj(
            bucket,
            object_name,
            buffer,
            content_type=content_type,
            metadata={"Content-Type": content_type},
        )
    def get_presigned_url(
        self,
        bucket: str,
        object_name: str,
        *,
        expires_seconds: timedelta = timedelta(seconds=3600),
    ) -> str:
        self._ensure_bucket(bucket)
        try:
            url = self.client.presigned_get_object(bucket, object_name, expires=expires_seconds)
            logger.debug("Generated presigned URL for %s/%s", bucket, object_name)
            return url
        except S3Error as exc:
            logger.exception("Failed to generate presigned URL for %s/%s", bucket, object_name)
            raise StorageError(f"Failed to presign object {bucket}/{object_name}: {exc}") from exc

    def list_objects(self, bucket: str, prefix: Optional[str] = None) -> Iterable[str]:
        self._ensure_bucket(bucket)
        try:
            for obj in self.client.list_objects(bucket, prefix=prefix or "", recursive=True):
                if obj.object_name is not None:
                    yield obj.object_name
        except S3Error as exc:
            logger.exception("Failed to list objects for %s/%s", bucket, prefix)
            raise StorageError(f"Failed to list objects: {exc}") from exc

    def get_object(self, bucket: str, object_name: str) -> bytes | None:
        self._ensure_bucket(bucket)
        try:
            response = self.client.get_object(bucket, object_name)
            try:
                data = response.read()
                return data
            finally:
                response.close()
                response.release_conn()
        except S3Error as exc:
            logger.info(f"Bucket: {bucket} has no {object_name}")
            return None
    def read_json(self, bucket: str, object_name: str) -> Dict[str, Any] | None:
        raw = self.get_object(bucket, object_name)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise StorageError(f"Stored object {bucket}/{object_name} is not valid JSON: {exc}") from exc

    def object_exists(self, bucket:str, object_name: str) -> bool:
        self._ensure_bucket(bucket)
        try:
            self.client.stat_object(bucket, object_name)
            return True
        except S3Error as exc:
            if exc.code in ("NoSuchKey", "NoSuchObject"):
                return False
            raise StorageError(f"Error checking object {bucket}/{object_name}: {exc}") from exc



__all__ = ["StorageClient", "StorageError"]
