"""OCR Elasticsearch indexing task."""

from __future__ import annotations

from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact, acreate_table_artifact

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import ImageOCRArtifact
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import get_postgres_client, shutdown_postgres_client
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.elasticsearch import ElasticsearchOCRClient, ElasticsearchSettings
from video_pipeline.core.client.inference.te_client import MMBertClient, MMBertConfig
from video_pipeline.config import get_settings


OCR_INDEXING_CONFIG = TaskConfig.from_yaml("ocr_indexing")
_base_kwargs = OCR_INDEXING_CONFIG.to_task_kwargs()


@StageRegistry.register
class OCRIndexingTask(BaseTask[list[ImageOCRArtifact], list[str]]):
    """Index OCR text documents into Elasticsearch for text search retrieval.

    preprocess() extracts OCR text from artifacts.
    execute() indexes documents into Elasticsearch with optional embeddings.
    postprocess() returns indexed artifact IDs.
    """

    config = OCR_INDEXING_CONFIG

    async def preprocess(self, input_data: list[ImageOCRArtifact]) -> list[dict]:
        """Extract OCR text and metadata from artifacts."""
        logger = get_run_logger()
        logger.info(f"[OCRIndexingTask] Preparing {len(input_data)} OCR document(s)")

        documents = []
        for artifact in input_data:
            ocr_text = (artifact.metadata or {}).get("ocr_text", "")
            if not ocr_text.strip():
                logger.debug(f"[OCRIndexingTask] Skipping empty OCR for frame {artifact.frame_index}")
                continue

            documents.append({
                "artifact_id": artifact.artifact_id,
                "user_id": artifact.user_id,
                "raw_text": ocr_text,
                "frame_index": artifact.frame_index,
                "timestamp": artifact.timestamp,
                "timestamp_sec": artifact.timestamp_sec,
                "video_id": artifact.related_video_id,
                "related_video_fps": artifact.related_video_fps,
                "image_minio_url": artifact.image_minio_url,
                "image_id": artifact.image_id,
            })

        logger.info(f"[OCRIndexingTask] Preprocessing done — {len(documents)} document(s) ready")
        return documents

    async def execute(self, preprocessed: list[dict], client: ElasticsearchOCRClient) -> list[str]:
        """Index OCR documents into Elasticsearch."""
        logger = get_run_logger()
        logger.info(f"[OCRIndexingTask] Indexing {len(preprocessed)} OCR document(s)")

        await client.create_index_if_not_exists()
        indexed_count = await client.batch_index_ocr_documents(
            documents=preprocessed,
            generate_embeddings=True,
        )

        artifact_ids = [doc["artifact_id"] for doc in preprocessed]
        logger.info(f"[OCRIndexingTask] Indexed {indexed_count} document(s) into Elasticsearch")
        return artifact_ids

    async def postprocess(self, result: list[str]) -> list[str]:
        """Return indexed artifact IDs."""
        return result

    @staticmethod
    async def summary_artifact(final_result: list[str]) -> None:
        """Create a summary artifact for OCR indexing."""
        if not final_result:
            return

        key = f"ocr-indexing-{len(final_result)}"
        markdown = (
            f"# OCR Indexing Summary\n\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Documents Indexed** | `{len(final_result)}` |\n"
        )

        await acreate_markdown_artifact(
            key=key,
            markdown=markdown,
            description="OCR indexing summary",
        )

        await acreate_table_artifact(
            table=[{"Field": "Documents Indexed", "Value": str(len(final_result))}],
            key=f"{key}-table",
            description="OCR indexing stats",
        )


@task(**{**_base_kwargs, "name": "OCR Indexing Chunk"})  # type: ignore
async def ocr_indexing_chunk_task(items: list[ImageOCRArtifact]) -> list[str]:
    """Index a batch of OCR artifacts into Elasticsearch.

    Args:
        items: Batch of ImageOCRArtifacts to index.

    Returns:
        List of indexed artifact IDs.
    """
    logger = get_run_logger()
    settings = get_settings()
    logger.info(f"[OCRIndexingChunk] Starting | {len(items)} artifact(s) in batch")

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()

    es_settings = ElasticsearchSettings(**settings.elasticsearch.model_dump())
    logger.info(f"[OCRIndexingChunk] Elasticsearch config | host={es_settings.host}:{es_settings.port}")

    embedding_base_url = OCR_INDEXING_CONFIG.additional_kwargs.get(
        "embedding_base_url", "http://mmbert:8000"
    )
    mmbert_config = MMBertConfig(base_url=embedding_base_url)
    mmbert_client = MMBertClient(config=mmbert_config)

    es_client = ElasticsearchOCRClient(settings=es_settings, embedding_client=mmbert_client)
    await es_client.connect()

    task_impl = OCRIndexingTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
    )

    try:
        indexed_ids = await task_impl.execute_template(items, es_client)
    finally:
        await es_client.close()
        await mmbert_client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(f"[OCRIndexingChunk] Done | {len(indexed_ids)} document(s) indexed")
    return indexed_ids