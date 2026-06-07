from __future__ import annotations

import os
from typing import Any

from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact
from pydantic import SecretStr

from video_pipeline.config import get_settings
from video_pipeline.core.artifact import KGExtractionArtifact, KGGraphArtifact, SegmentCaptionArtifact
from video_pipeline.core.client.inference import MMBertClient, MMBertConfig, SpladeClient, SpladeConfig
from video_pipeline.core.client.llm_provider.openrouter import OpenRouterClient, OpenRouterConfig
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import get_postgres_client, shutdown_postgres_client
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.task.base.base_task import BaseTask, TaskConfig

from .entity_resolution import run_entity_resolution
from .event_linking import run_event_linking
from .extract_kg import caption_segment_from_artifact, extract_kg_graph
from .models import CaptionSegment, CostTracker, KGEntityResolutionResult, KGSegment


KG_EXTRACTION_CONFIG = TaskConfig.from_yaml("kg_extraction")
KG_ENTITY_RESOLUTION_CONFIG = TaskConfig.from_yaml("kg_entity_resolution")
KG_FINALIZATION_CONFIG = TaskConfig.from_yaml("kg_finalization")


def _make_llm_client(kwargs: dict[str, Any]) -> OpenRouterClient:
    return OpenRouterClient(
        config=OpenRouterConfig(
            model=kwargs.get("model", "qwen/qwen3-coder-next"),
            base_url=kwargs.get("base_url", "https://openrouter.ai/api/v1"),
            max_tokens=kwargs.get("max_tokens", 8192),
            reasoning_effort=kwargs.get("reasoning_effort", "none"),
            api_key=SecretStr(os.environ.get("OPENROUTER_API_KEY", "")),
        )
    )


def _new_cost_tracker(kwargs: dict[str, Any]) -> CostTracker:
    return CostTracker(model=kwargs.get("model", "qwen/qwen3-coder-next"))


def _tracker_from_extractions(
    extraction_artifacts: list[KGExtractionArtifact],
    kwargs: dict[str, Any],
) -> CostTracker:
    tracker = _new_cost_tracker(kwargs)
    for artifact in extraction_artifacts:
        tracker.total_prompt_tokens += artifact.total_prompt_tokens
        tracker.total_completion_tokens += artifact.total_completion_tokens
        tracker.total_cost += artifact.total_llm_cost
        tracker.llm_calls += artifact.llm_calls
    return tracker


@StageRegistry.register
class KGExtractionTask(BaseTask[list[SegmentCaptionArtifact], KGExtractionArtifact]):
    """Extract raw per-segment KG payloads for one segment-caption chunk."""

    config = KG_EXTRACTION_CONFIG

    async def preprocess(self, input_data: list[SegmentCaptionArtifact]) -> list[CaptionSegment]:
        logger = get_run_logger()
        logger.info(f"[KGExtraction] Preprocessing {len(input_data)} segment(s)")
        return [caption_segment_from_artifact(a) for a in input_data]

    async def execute(
        self,
        preprocessed: list[CaptionSegment],
        client: dict[str, Any],
    ) -> tuple[list[KGSegment], CostTracker]:
        logger = get_run_logger()
        cost_tracker = _new_cost_tracker(self.config.additional_kwargs)
        kg_segments = await extract_kg_graph(
            preprocessed,
            client["llm"],
            max_concurrent=self.config.additional_kwargs.get("max_concurrent", 5),
            cost_tracker=cost_tracker,
        )
        logger.info(f"[KGExtraction] Extracted KG from {len(kg_segments)} segment(s)")
        return kg_segments, cost_tracker

    async def postprocess(
        self,
        result: tuple[list[KGSegment], CostTracker],
    ) -> KGExtractionArtifact:
        kg_segments, cost_tracker = result
        artifact = KGExtractionArtifact(
            user_id=self.kwargs.get("user_id", "unknown"),
            related_video_id=self.kwargs.get("video_id", "unknown"),
            related_segment_caption_artifact_ids=self.kwargs.get("segment_caption_artifact_ids", []),
            kg_segments=[segment.model_dump() for segment in kg_segments],
            total_raw_entities=sum(len(segment.entities) for segment in kg_segments),
            total_prompt_tokens=cost_tracker.total_prompt_tokens,
            total_completion_tokens=cost_tracker.total_completion_tokens,
            total_llm_cost=cost_tracker.total_cost,
            llm_model=cost_tracker.model,
            llm_calls=cost_tracker.llm_calls,
        )
        if self.artifact_visitor:
            await self.artifact_visitor.visit_artifact(artifact)
        return artifact

    @staticmethod
    async def summary_artifact(final_result: KGExtractionArtifact) -> None:
        return None


@StageRegistry.register
class KGPipelineTask(BaseTask[KGEntityResolutionResult, KGGraphArtifact]):
    """Finalize a resolved KG into the graph artifact consumed downstream."""

    config = KG_FINALIZATION_CONFIG

    async def preprocess(self, input_data: KGEntityResolutionResult) -> KGEntityResolutionResult:
        return input_data

    async def execute(
        self,
        preprocessed: KGEntityResolutionResult,
        client: dict[str, Any],
    ) -> KGGraphArtifact:
        logger = get_run_logger()
        kwargs = self.config.additional_kwargs
        cost_tracker = preprocessed.cost_tracker
        enhanced_kg = await run_event_linking(
            preprocessed.resolved_kg,
            client["dense"],
            client["llm"],
            semantic_threshold=kwargs.get("semantic_threshold", 0.80),
            llm_confirm_threshold=kwargs.get("llm_confirm_threshold", 0.60),
            jaccard_threshold=kwargs.get("jaccard_threshold", 0.30),
            micro_window_size=kwargs.get("micro_window_size", 2),
            micro_semantic_threshold=kwargs.get("micro_semantic_threshold", 0.85),
            micro_llm_confirm_threshold=kwargs.get("micro_llm_confirm_threshold", 0.65),
            micro_jaccard_threshold=kwargs.get("micro_jaccard_threshold", 0.40),
            max_concurrent_llm=kwargs.get("max_concurrent_llm", 5),
            cost_tracker=cost_tracker,
        )
        logger.info(
            f"[KGFinalization] Built {len(enhanced_kg.events)} events, "
            f"{len(enhanced_kg.micro_event_nodes)} micro-events"
        )
        return KGGraphArtifact(
            user_id=self.kwargs.get("user_id", "unknown"),
            related_video_id=enhanced_kg.video_id,
            related_segment_caption_artifact_ids=preprocessed.related_segment_caption_artifact_ids,
            related_kg_extraction_artifact_ids=preprocessed.related_kg_extraction_artifact_ids,
            entities=[e.model_dump() for e in enhanced_kg.entities],
            relationships=[r.model_dump() for r in enhanced_kg.relationships],
            segment_views=[s.model_dump() for s in enhanced_kg.segments],
            events=[e.to_raw_dict() for e in enhanced_kg.events],
            event_entity_links=[e.to_arango_doc() for e in enhanced_kg.event_entity_links],
            event_edges=[e.to_arango_doc() for e in enhanced_kg.event_edges],
            micro_event_nodes=[m.to_raw_dict() for m in enhanced_kg.micro_event_nodes],
            micro_event_edges=[e.to_arango_doc() for e in enhanced_kg.micro_event_edges],
            total_raw_entities=preprocessed.total_raw_entities,
            total_canonical_entities=len(enhanced_kg.entities),
            total_relationships=len(enhanced_kg.relationships),
            total_events=len(enhanced_kg.events),
            total_micro_events=len(enhanced_kg.micro_event_nodes),
            total_event_edges=len(enhanced_kg.event_edges),
            total_micro_event_edges=len(enhanced_kg.micro_event_edges),
            total_prompt_tokens=cost_tracker.total_prompt_tokens,
            total_completion_tokens=cost_tracker.total_completion_tokens,
            total_llm_cost=cost_tracker.total_cost,
            llm_model=cost_tracker.model,
            llm_calls=cost_tracker.llm_calls,
        )

    async def postprocess(self, result: KGGraphArtifact) -> KGGraphArtifact:
        if self.artifact_visitor:
            await self.artifact_visitor.visit_artifact(result)
        return result

    @staticmethod
    async def summary_artifact(final_result: KGGraphArtifact) -> None:
        if not final_result.entities:
            return
        canonical_count = final_result.total_canonical_entities
        ratio = (
            f"{final_result.total_raw_entities / canonical_count:.2f}x"
            if canonical_count
            else "N/A"
        )
        cost_str = f"${final_result.total_llm_cost:.4f}" if final_result.total_llm_cost > 0 else "N/A"
        tokens_str = f"{final_result.total_prompt_tokens:,}" if final_result.total_prompt_tokens > 0 else "N/A"
        markdown = (
            f"# Knowledge Graph Pipeline Summary\n\n"
            f"## Video Information\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| **Video ID** | `{final_result.related_video_id}` |\n"
            f"| **Model Used** | `{final_result.llm_model or 'N/A'}` |\n\n"
            f"## Entity Statistics\n"
            f"| Field | Count |\n|-------|-------|\n"
            f"| **Raw Entities (extracted)** | `{final_result.total_raw_entities}` |\n"
            f"| **Canonical Entities (resolved)** | `{canonical_count}` |\n"
            f"| **Entity Resolution Ratio** | `{ratio}` |\n"
            f"| **Global Relationships** | `{final_result.total_relationships}` |\n\n"
            f"## Event Layer\n"
            f"| Field | Count |\n|-------|-------|\n"
            f"| **Big Events** | `{final_result.total_events}` |\n"
            f"| **Micro-Events** | `{final_result.total_micro_events}` |\n"
            f"| **Event-to-Event Edges** | `{final_result.total_event_edges}` |\n"
            f"| **Micro-Event Edges** | `{final_result.total_micro_event_edges}` |\n\n"
            f"## Cost & Usage\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| **LLM Calls** | `{final_result.llm_calls}` |\n"
            f"| **Total Tokens (Prompt)** | `{tokens_str}` |\n"
            f"| **Estimated Cost** | `{cost_str}` |\n"
        )
        await acreate_markdown_artifact(
            key=f"kg-pipeline-{final_result.related_video_id}".lower(),
            markdown=markdown,
            description=f"KG pipeline summary for video {final_result.related_video_id}",
        )


@task(**{**KG_EXTRACTION_CONFIG.to_task_kwargs(), "name": "KG Extraction Chunk"})  # type: ignore
async def kg_extraction_chunk_task(
    segments: list[SegmentCaptionArtifact],
) -> KGExtractionArtifact:
    logger = get_run_logger()
    settings = get_settings()
    kwargs = KG_EXTRACTION_CONFIG.additional_kwargs
    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()
    llm_client = _make_llm_client(kwargs)
    task_impl = KGExtractionTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
        user_id=segments[0].user_id if segments else "unknown",
        video_id=segments[0].related_video_id if segments else "unknown",
        segment_caption_artifact_ids=[segment.artifact_id for segment in segments],
    )
    try:
        result = await task_impl.execute_template(segments, {"llm": llm_client})
    finally:
        await llm_client.close()
        await shutdown_postgres_client(postgres_client)
    logger.info(f"[KGExtraction] Done | artifact {result.artifact_id}")
    return result


@task(**{**KG_ENTITY_RESOLUTION_CONFIG.to_task_kwargs(), "name": "KG Entity Resolution"})  # type: ignore
async def kg_entity_resolution_task(
    extraction_artifacts: list[KGExtractionArtifact],
) -> KGEntityResolutionResult:
    if not extraction_artifacts:
        raise ValueError("At least one KG extraction artifact is required")
    logger = get_run_logger()
    kwargs = KG_ENTITY_RESOLUTION_CONFIG.additional_kwargs
    llm_client = _make_llm_client(kwargs)
    dense_client = MMBertClient(MMBertConfig(base_url=kwargs.get("dense_embedding_base_url", "http://mmbert:8000")))
    sparse_client = SpladeClient(SpladeConfig(url=kwargs.get("sparse_embedding_url", "triton:8001")))
    kg_segments = [
        KGSegment.model_validate(segment)
        for artifact in extraction_artifacts
        for segment in artifact.kg_segments
    ]
    kg_segments.sort(key=lambda x: (x.from_batch, x.to_batch))
    cost_tracker = _tracker_from_extractions(extraction_artifacts, kwargs)
    try:
        resolved_kg = await run_entity_resolution(
            kg_segments,
            llm_client,
            dense_client,
            sparse_client,
            extraction_artifacts[0].related_video_id,
            dense_weight=kwargs.get("hybrid_dense_weight", 0.9),
            sparse_weight=kwargs.get("hybrid_sparse_weight", 0.1),
            sim_threshold=kwargs.get("similarity_threshold", 0.75),
            max_concurrent=kwargs.get("max_concurrent_llm", 5),
            max_entities_per_cluster=kwargs.get("max_entities_per_cluster", 25),
            cost_tracker=cost_tracker,
        )
    finally:
        await llm_client.close()
        await dense_client.close()
    logger.info(f"[KGEntityResolution] Resolved to {len(resolved_kg.entities)} canonical entities")
    return KGEntityResolutionResult(
        resolved_kg=resolved_kg,
        cost_tracker=cost_tracker,
        user_id=extraction_artifacts[0].user_id,
        related_segment_caption_artifact_ids=[
            artifact_id
            for artifact in extraction_artifacts
            for artifact_id in artifact.related_segment_caption_artifact_ids
        ],
        related_kg_extraction_artifact_ids=[artifact.artifact_id for artifact in extraction_artifacts],
        total_raw_entities=sum(artifact.total_raw_entities for artifact in extraction_artifacts),
    )


@task(**{**KG_FINALIZATION_CONFIG.to_task_kwargs(), "name": "KG Finalization"})  # type: ignore
async def kg_finalization_task(
    resolution_result: KGEntityResolutionResult,
) -> KGGraphArtifact:
    settings = get_settings()
    kwargs = KG_FINALIZATION_CONFIG.additional_kwargs
    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()
    llm_client = _make_llm_client(kwargs)
    dense_client = MMBertClient(MMBertConfig(base_url=kwargs.get("dense_embedding_base_url", "http://mmbert:8000")))
    task_impl = KGPipelineTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
        user_id=resolution_result.user_id,
    )
    try:
        result = await task_impl.execute_template(
            resolution_result,
            {"llm": llm_client, "dense": dense_client},
        )
    finally:
        await llm_client.close()
        await dense_client.close()
        await shutdown_postgres_client(postgres_client)
    return result
