import urllib3
from urllib3.util import Timeout
from minio import Minio
from minio.error import S3Error
from typing import Dict, Any, BinaryIO
import json


<<<<<<< HEAD
import logging


=======
>>>>>>> upstream/feat/agent
class StorageError(RuntimeError):
    """Raised when MinIO storage operations fail."""
    pass

class StorageClient:
    def __init__(self, host:str, port: str, access_key:str, secret_key:str,secure:bool) -> None:
        timeout = Timeout(connect=5.0, read=120.0)  
        self._http_client = urllib3.PoolManager(
            maxsize=50,
            timeout=timeout,
            block=True,
        )
        self.client = Minio(
            endpoint=f"{host}:{port}",
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            http_client=self._http_client,
        )

        self.logger = logging.getLogger(__name__)
    
    def _ensure_bucket(self, bucket: str) -> None:
        try:
            if not self.client.bucket_exists(bucket):
<<<<<<< HEAD
                self.logger.info(f"Bucket named: {bucket} does not exist, creating")
                self.client.make_bucket(bucket)
        except S3Error as exc:
            self.logger.error("MinIO bucket check failed for %s: %s", bucket, exc)
=======
                self.client.make_bucket(bucket)
        except S3Error as exc:
>>>>>>> upstream/feat/agent
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
<<<<<<< HEAD
            self.logger.info(f"Bucket: {bucket} has no {object_name}")
=======
>>>>>>> upstream/feat/agent
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

    