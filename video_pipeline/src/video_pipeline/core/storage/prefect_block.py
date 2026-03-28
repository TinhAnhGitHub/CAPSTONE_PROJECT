"""Prefect storage blocks for result persistence using MinIO (S3-compatible)."""

from pydantic import Field, SecretStr
from prefect.blocks.core import Block


def create_minio_result_storage(
    endpoint: str = "minio:9000",
    access_key: str = "minioadmin",
    secret_key: str = "minioadmin",
    bucket: str = "prefect-results",
    secure: bool = False,
) -> "S3Bucket": #type:ignore
    """Create an S3Bucket block configured for MinIO.

    This creates a Prefect S3Bucket storage block that works with MinIO
    by setting the custom endpoint_url.

    Args:
        endpoint: MinIO endpoint (host:port)
        access_key: MinIO access key
        secret_key: MinIO secret key
        bucket: Bucket name for storing results
        secure: Use HTTPS

    Returns:
        S3Bucket block configured for MinIO
    """
    from prefect_aws.s3 import S3Bucket
    from prefect_aws import AwsCredentials
    from prefect_aws.client_parameters import AwsClientParameters

    
    protocol = "https" if secure else "http"
    endpoint_url = f"{protocol}://{endpoint}"

    credentials = AwsCredentials(
        aws_access_key_id=access_key,
        aws_secret_access_key=SecretStr(secret_key),
        region_name="us-east-1",
        aws_client_parameters=AwsClientParameters(
            endpoint_url=endpoint_url
        ),
    )

    s3_bucket = S3Bucket(
        bucket_name=bucket,
        credentials=credentials,
    )

    return s3_bucket


class MinIOStorageBlock(Block):
    """MinIO storage block for custom storage operations.

    This is a simpler block for custom MinIO operations that don't need
    to implement the full Prefect result storage interface.

    For Prefect result persistence, use create_minio_result_storage() instead.
    """
    _block_type_name = "MinIO Storage"
    _block_type_slug = "minio-storage"
    _description = "MinIO S3-compatible storage for custom operations."

    endpoint: str = Field(
        default="minio:9000",
        description="MinIO endpoint URL (host:port)"
    )
    access_key: SecretStr = Field(
        default=SecretStr("minioadmin"),
        description="MinIO access key"
    )
    secret_key: SecretStr = Field(
        default=SecretStr("minioadmin"),
        description="MinIO secret key"
    )
    bucket: str = Field(
        default="prefect-results",
        description="Bucket name for storing results"
    )
    secure: bool = Field(
        default=False,
        description="Use HTTPS for MinIO connection"
    )

    class Config:
        populate_by_name = True

    def get_client(self):
        """Get a MinIO client instance."""
        from minio import Minio
        return Minio(
            self.endpoint,
            access_key=self.access_key.get_secret_value(),
            secret_key=self.secret_key.get_secret_value(),
            secure=self.secure
        )

    async def write(self, key: str, data: bytes) -> str:
        """Write data to MinIO.

        Args:
            key: Object key/path within the bucket
            data: Bytes to write

        Returns:
            Full path to the stored object
        """
        from io import BytesIO
        client = self.get_client()
        client.put_object(
            self.bucket,
            key,
            BytesIO(data),
            length=len(data)
        )
        return f"{self.bucket}/{key}"

    async def read(self, key: str) -> bytes:
        """Read data from MinIO.

        Args:
            key: Object key/path within the bucket

        Returns:
            Object data as bytes
        """
        client = self.get_client()
        response = client.get_object(self.bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def ensure_bucket_exists(self) -> bool:
        """Create bucket if it doesn't exist.

        Returns:
            True if bucket exists or was created
        """
        client = self.get_client()
        if client.bucket_exists(self.bucket):
            return True
        client.make_bucket(self.bucket)
        return True