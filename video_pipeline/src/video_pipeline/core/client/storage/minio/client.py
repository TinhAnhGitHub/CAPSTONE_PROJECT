from typing import Any, BinaryIO
import json
from io import BytesIO
from minio import Minio
from minio.error import S3Error
import urllib3
from urllib3.util import Timeout
from loguru import logger

from .exception import MinioStorageError
from .config import MinioConfig


class MinioStorageClient:
    def __init__(self, config: MinioConfig):
        self.config = config
        timeout = Timeout(connect=5.0, read=20.0)
        self._http_client = urllib3.PoolManager(
            maxsize=200,
            timeout=timeout,
            block=False,
        )
        self.client = Minio(
            endpoint=f"{config.host}:{config.port}",
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=config.secure,
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

        put_kwargs: dict[str, Any] = dict(
            bucket_name=bucket,
            object_name=object_name,
            data=file_obj,
            content_type=content_type,
            metadata=metadata,
        )

        try:
            self.client.put_object(**put_kwargs)
            uri = f"s3://{bucket}/{object_name}"
            logger.info(f"Uploaded object {uri}")
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
