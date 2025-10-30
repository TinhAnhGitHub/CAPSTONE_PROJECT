from ingestion.core.config.storage import MinioSettings
import urllib3
from urllib3.util import Timeout
from minio import Minio
from minio.error import S3Error
from typing import Dict, Any, BinaryIO
import json


class StorageError(RuntimeError):
    """Raised when MinIO storage operations fail."""
    pass

class StorageClient:
    def __init__(self, settings: MinioSettings) -> None:
        self.settings = settings 
        timeout = Timeout(connect=5.0, read=120.0)  
        self._http_client = urllib3.PoolManager(
            maxsize=50,
            timeout=timeout,
            block=True,
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
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
        except S3Error as exc:
            raise StorageError(f"Failed to ensure bucket {bucket}: {exc}") from exc

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
            return None

    def read_json(self, bucket: str, object_name: str) -> Dict[str, Any] | None:
        raw = self.get_object(bucket, object_name)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise StorageError(f"Stored object {bucket}/{object_name} is not valid JSON: {exc}") from exc
    
    def upload_fileobj(
        self,
        bucket: str,
        object_name: str,
        file_obj: BinaryIO,
        *,
        content_type: str = "application/octet-stream",
        metadata: Dict[str, str] | None = None,
    ) -> str:
        """Upload a file-like object to the specified bucket and return an s3 URI."""

        self._ensure_bucket(bucket)
        try:
            self.client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=file_obj,
                length=-1,
                part_size=10 * 1024 * 1024,
                content_type=content_type,
                metadata=metadata, #type:ignore
            )
            uri = f"s3://{bucket}/{object_name}"
            return uri
        except S3Error as exc:
            raise StorageError(f"Upload failed for {object_name}: {exc}") from exc

    