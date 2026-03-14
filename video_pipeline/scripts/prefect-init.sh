#!/usr/bin/env bash
set -e

echo "Waiting for Prefect API..."
until python - <<EOF
import urllib.request
urllib.request.urlopen("http://prefect-server:4200/api/health")
EOF
do
  sleep 2
done

echo "Prefect API ready"
echo "Ensuring work pool exists..."
prefect work-pool create --type process local-pool || true

echo "Ensuring concurrency limits..."
prefect concurrency-limit create llm-task 5 || true
prefect concurrency-limit create embedding-task 50 || true
prefect concurrency-limit create image-chunk-task 10 || true
# prefect concurrency-limit create autoshot-task 20 || true
# # prefect concurrency-limit create asr-task 20 || true
# prefect concurrency-limit create video-registry 1 || true

echo "Creating MinIO result storage block..."
python << 'PYTHON_EOF'
import sys
try:
    from fastembed import SparseTextEmbedding
    from video_pipeline.core.storage.prefect_block import create_minio_result_storage
    from prefect_aws.s3 import S3Bucket

    model = SparseTextEmbedding(
        model_name="prithivida/Splade_PP_en_v1",
        cache_dir="/models/fastembed"
    )
    print("SPLADE model downloaded and cached")

    # Check if block already exists
    try:
        existing = S3Bucket.load("result-storage")
        print(f"Block 'result-storage' already exists with bucket: {existing.bucket_name}")
    except ValueError:
        # Create new S3Bucket block configured for MinIO
        s3_bucket = create_minio_result_storage(
            endpoint="minio:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            bucket="prefect-results",
            secure=False
        )
        s3_bucket.save("result-storage", overwrite=True)
        print("Result storage block 'result-storage' created successfully")

    # Ensure bucket exists in MinIO
    from video_pipeline.core.storage.prefect_block import MinIOStorageBlock
    minio_block = MinIOStorageBlock.load("result-storage") if False else MinIOStorageBlock(
        endpoint="minio:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        bucket="prefect-results",
        secure=False
    )
    minio_block.ensure_bucket_exists()
    print(f"Bucket 'prefect-results' verified")

except Exception as e:
    print(f"Warning: Could not create storage block: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
PYTHON_EOF

echo "Deploying flows..."
cd /app
prefect --no-prompt deploy --all

echo "Prefect initialization complete"