from typing import Any, BinaryIO
import json
from contextlib import asynccontextmanager
import tempfile
import asyncio
import os
from urllib.parse import urlparse
from io import BytesIO
from minio import Minio
from minio.error import S3Error
import urllib3
from urllib3.util import Timeout
from loguru import logger

from .exception import MinioStorageError


class MinioStorageClient:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, secure: bool):
        timeout = Timeout(connect=5.0, read=20.0)
        self._http_client = urllib3.PoolManager(
            maxsize=200,
            timeout=timeout,
            block=False,
        )
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
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
            raise MinioStorageError(f"Failed to ensure bucket {bucket}: {exc}") from exc

    def make_bucket_public(self, bucket: str) -> None:
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{bucket}/*"],
                }
            ],
        }
        self.client.set_bucket_policy(bucket_name=bucket, policy=json.dumps(policy))

    def upload_fileobj(
        self,
        bucket: str,
        object_name: str,
        file_obj: BinaryIO,
        *,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        self._ensure_bucket(bucket)

        current_pos = file_obj.tell()
        file_obj.seek(0, os.SEEK_END)
        length = file_obj.tell()
        file_obj.seek(current_pos)


        try:
            self.client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=file_obj,
                length=length,
                content_type=content_type,
                metadata=metadata, #type:ignore
            )
            uri = f"s3://{bucket}/{object_name}"
            return uri
        except S3Error as exc:
            logger.exception("Failed to upload %s to bucket %s", object_name, bucket)
            raise MinioStorageError(f"Upload failed for {object_name}: {exc}") from exc

    def upload_json_file(
        self,
        bucket: str,
        object_name: str,
        payload: dict[str, Any],
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

    def object_exists(self, bucket: str, object_name: str) -> bool:
        self._ensure_bucket(bucket)
        try:
            self.client.stat_object(bucket, object_name)
            return True
        except S3Error as exc:
            if exc.code in ("NoSuchKey", "NoSuchObject"):
                return False
            raise MinioStorageError(
                f"Error checking object {bucket}/{object_name}: {exc}"
            ) from exc

    def get_object_bytes(self, bucket: str, object_name: str) -> bytes:
        try:
            response = self.client.get_object(bucket, object_name)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except S3Error as exc:
            logger.exception("Failed to fetch %s/%s", bucket, object_name)
            raise MinioStorageError(
                f"Failed to fetch object {bucket}/{object_name}: {exc}"
            ) from exc

    def stream_object_to_file(self, bucket: str, object_name: str, file_path: str, chunk_size: int = 32 * 1024 * 1024) -> int:
        """Stream object from MinIO directly to file without loading all into memory.

        Args:
            bucket: Bucket name
            object_name: Object name
            file_path: Path to write the file
            chunk_size: Size of chunks to read at a time (default 32MB)

        Returns:
            Total bytes written
        """
        try:
            response = self.client.get_object(bucket, object_name)
            try:
                total_bytes = 0
                with open(file_path, 'wb') as f:
                    for chunk in response.stream(chunk_size):
                        f.write(chunk)
                        total_bytes += len(chunk)
                return total_bytes
            finally:
                response.close()
                response.release_conn()
        except S3Error as exc:
            logger.exception("Failed to stream %s/%s", bucket, object_name)
            raise MinioStorageError(
                f"Failed to stream object {bucket}/{object_name}: {exc}"
            ) from exc


    @asynccontextmanager
    async def fetch_object_from_s3(self, s3_url: str, suffix: str):
        """Fetch object from S3 into memory first, then write to temp file.

        DEPRECATED: Use fetch_object_streaming() for memory efficiency.
        This method loads the entire object into memory before writing to disk.
        """
        parsed = urlparse(s3_url)
        if parsed.scheme == "s3":
            # s3://bucket/object
            bucket = parsed.netloc
            object_name = parsed.path.lstrip("/")
        else:
            # http(s)://host:port/bucket/object
            path_parts = parsed.path.lstrip("/").split("/", 1)
            bucket = path_parts[0]
            object_name = path_parts[1] if len(path_parts) > 1 else ""

        loop = asyncio.get_running_loop()

        data = await loop.run_in_executor(
            None,
            lambda: self.get_object_bytes(bucket, object_name),
        )

        if data is None:
            raise FileNotFoundError(f"Object {s3_url} not found in storage")

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)

        try:
            tmp.write(data)
            tmp.flush()
            tmp.close()
            yield tmp.name
        finally:
            try:
                os.remove(tmp.name)
            except FileNotFoundError:
                pass

    @asynccontextmanager
    async def fetch_object_streaming(self, s3_url: str, suffix: str):
        """Stream object from MinIO directly to temp file without loading into memory.

        This is the memory-efficient version of fetch_object_from_s3.
        Instead of loading the entire object into RAM before writing to disk,
        it streams chunks directly to a temp file.

        Args:
            s3_url: S3 URL (s3://bucket/object) or HTTP URL
            suffix: File suffix for temp file (e.g., '.mp4')

        Yields:
            Path to temp file (deleted after context exits)
        """
        parsed = urlparse(s3_url)
        if parsed.scheme == "s3":
            bucket = parsed.netloc
            object_name = parsed.path.lstrip("/")
        else:
            path_parts = parsed.path.lstrip("/").split("/", 1)
            bucket = path_parts[0]
            object_name = path_parts[1] if len(path_parts) > 1 else ""

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()  # Close so we can write to it separately

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.stream_object_to_file(bucket, object_name, tmp.name)
            )
            yield tmp.name
        finally:
            try:
                os.remove(tmp.name)
            except FileNotFoundError:
                pass
    
    def delete_object(self, bucket: str, object_name: str) -> None:
        self._ensure_bucket(bucket)

        try:
            self.client.remove_object(bucket, object_name)
            logger.info(f"Delete {object_name=} in {bucket=}")
        except S3Error as exc:
            logger.exception(f"Failed to delete {object_name} from {bucket}")
            raise MinioStorageError(
                f"Failed to delete object {bucket}/{object_name}: {exc}"
            ) from exc
    