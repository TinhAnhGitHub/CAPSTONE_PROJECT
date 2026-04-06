from __future__ import annotations

import os
from typing import Any

from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact
from pydantic import SecretStr

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import SegmentCaptionArtifact, KGGraphArtifact
from video_pipeline.core.storage.pg_tracker import ArtifactPersistentVisitor
from video_pipeline.core.client.storage.minio import MinioStorageClient
from video_pipeline.core.client.storage.pg.runtime import get_postgres_client, shutdown_postgres_client
from video_pipeline.core.client.llm_provider.openrouter import OpenRouterClient, OpenRouterConfig
from video_pipeline.core.client.inference import MMBertClient, MMBertConfig, SpladeClient, SpladeConfig
from video_pipeline.config import get_settings

from .models import CaptionSegment, CostTracker
from .extract_kg import extract_kg_graph, caption_segment_from_artifact
from .entity_resolution import run_entity_resolution
from .event_linking import run_event_linking
from .community_detection import run_community_detection
from .node2vec_embeddings import run_node2vec


KG_PIPELINE_CONFIG = TaskConfig.from_yaml("kg_pipeline")


@StageRegistry.register
class KGPipelineTask(BaseTask[list[SegmentCaptionArtifact], KGGraphArtifact]):
    """Run the complete Knowledge Graph pipeline."""

    config = KG_PIPELINE_CONFIG

    async def preprocess(
        self,
        input_data: list[SegmentCaptionArtifact],
    ) -> list[CaptionSegment]:
        """Convert artifacts to CaptionSegments."""
        logger = get_run_logger()
        logger.info(f"[KGPipeline] Preprocessing {len(input_data)} segment(s)")
        return [caption_segment_from_artifact(a) for a in input_data]

    async def execute(
        self,
        preprocessed: list[CaptionSegment],
        client: dict[str, Any],
    ) -> KGGraphArtifact:
        """Run the full KG pipeline."""
        logger = get_run_logger()
        kwargs = self.config.additional_kwargs

        llm_client = client["llm"]
        dense_client = client["dense"]
        sparse_client = client["sparse"]

        video_id = preprocessed[0].video_id if preprocessed else "unknown"
        user_id = self.kwargs.get("user_id", "unknown")
        segment_caption_artifact_ids = self.kwargs.get("segment_caption_artifact_ids", [])

        cost_tracker = CostTracker()
        cost_tracker.model = kwargs.get("model", "qwen/qwen3-coder-next")

        logger.info(f"[KGPipeline] Stage 1: KG Extraction")
        logger.info(f"{len(preprocessed) } segment(s) to process for KG extraction")
        kg_segments = await extract_kg_graph(
            preprocessed,
            llm_client,
            max_concurrent=kwargs.get("max_concurrent", 5),
            cost_tracker=cost_tracker,
        )
        logger.info(f"[KGPipeline] Extracted KG from {len(kg_segments)} segment(s)")

        resolved_kg = await run_entity_resolution(
            kg_segments,
            llm_client,
            dense_client,
            sparse_client,
            video_id,
            dense_weight=kwargs.get("hybrid_dense_weight", 0.9),
            sparse_weight=kwargs.get("hybrid_sparse_weight", 0.1),
            sim_threshold=kwargs.get("similarity_threshold", 0.75),
            max_concurrent=kwargs.get("max_concurrent_llm", 5),
            cost_tracker=cost_tracker,
        )
        logger.info(f"[KGPipeline] Resolved to {len(resolved_kg.entities)} canonical entities")

        enhanced_kg = await run_event_linking(
            resolved_kg,
            dense_client,
            llm_client,
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
        logger.info(f"[KGPipeline] Built {len(enhanced_kg.events)} events, {len(enhanced_kg.micro_event_nodes)} micro-events")

        communities = await run_community_detection(
            enhanced_kg,
            dense_client,
            llm_client,
            n_iterations=kwargs.get("n_iterations", 10),
            seed=kwargs.get("seed", 42),
            max_concurrent_llm=kwargs.get("max_concurrent_llm", 5),
            cost_tracker=cost_tracker,
        )
        logger.info(f"[KGPipeline] Detected {len(communities.communities)} communities")

        node2vec_output = run_node2vec(
            enhanced_kg,
            communities,
            dim=kwargs.get("dim", 128),
            walk_length=kwargs.get("walk_length", 80),
            num_walks=kwargs.get("num_walks", 10),
            p=kwargs.get("p", 1.0),
            q=kwargs.get("q", 1.0),
            window=kwargs.get("window", 10),
            workers=kwargs.get("workers", 4),
            seed=kwargs.get("seed", 42),
        )
        logger.info(f"[KGPipeline] Generated embeddings for {len(node2vec_output.nodes)} nodes")

        logger.info(
            f"[KGPipeline] LLM Cost Summary | "
            f"calls={cost_tracker.llm_calls} | "
            f"prompt_tokens={cost_tracker.total_prompt_tokens:,} | "
            f"completion_tokens={cost_tracker.total_completion_tokens:,} | "
            f"cost=${cost_tracker.total_cost:.4f}"
        )

        total_raw_entities = sum(len(seg.entities) for seg in kg_segments)

        artifact = KGGraphArtifact(
            user_id=user_id,
            related_video_id=video_id,
            related_segment_caption_artifact_ids=segment_caption_artifact_ids,
            entities=[e.model_dump() for e in enhanced_kg.entities],
            relationships=[r.model_dump() for r in enhanced_kg.relationships],
            segment_views=[s.model_dump() for s in enhanced_kg.segments],
            events=[e.to_raw_dict() for e in enhanced_kg.events],
            event_entity_links=[e.to_arango_doc() for e in enhanced_kg.event_entity_links],
            event_edges=[e.to_arango_doc() for e in enhanced_kg.event_edges],
            micro_event_nodes=[m.to_raw_dict() for m in enhanced_kg.micro_event_nodes],
            micro_event_edges=[e.to_arango_doc() for e in enhanced_kg.micro_event_edges],
            communities=[c.model_dump() for c in communities.communities],
            membership_edges=[e.to_arango_doc() for e in communities.membership_edges],
            event_community_edges=[e.to_arango_doc() for e in communities.event_community_edges],
            node2vec_meta=node2vec_output.meta.model_dump(),
            node_embeddings={k: v.model_dump() for k, v in node2vec_output.nodes.items()},

            total_raw_entities=total_raw_entities,
            total_canonical_entities=len(enhanced_kg.entities),
            total_relationships=len(enhanced_kg.relationships),
            total_events=len(enhanced_kg.events),
            total_micro_events=len(enhanced_kg.micro_event_nodes),
            total_communities=len(communities.communities),
            total_event_edges=len(enhanced_kg.event_edges),
            total_micro_event_edges=len(enhanced_kg.micro_event_edges),
            total_membership_edges=len(communities.membership_edges),
            total_nodes_with_embeddings=len(node2vec_output.nodes),
            graph_modularity=communities.graph_stats.modularity,

            total_prompt_tokens=cost_tracker.total_prompt_tokens,
            total_completion_tokens=cost_tracker.total_completion_tokens,
            total_llm_cost=cost_tracker.total_cost,
            llm_model=cost_tracker.model,
            llm_calls=cost_tracker.llm_calls,
        )

        return artifact

    async def postprocess(
        self,
        result: KGGraphArtifact,
    ) -> KGGraphArtifact:
        """Persist the KG artifact to PostgreSQL."""
        logger = get_run_logger()

        if self.artifact_visitor:
            await self.artifact_visitor.visit_artifact(result)
            logger.info(f"[KGPipeline] Persisted KG artifact {result.artifact_id}")

        return result

    @staticmethod
    async def summary_artifact(final_result: KGGraphArtifact) -> None:
        """Create a Prefect artifact summary."""
        if not final_result.entities:
            return

        cost_str = f"${final_result.total_llm_cost:.4f}" if final_result.total_llm_cost > 0 else "N/A"
        tokens_str = f"{final_result.total_prompt_tokens:,}" if final_result.total_prompt_tokens > 0 else "N/A"

        markdown = (
            f"# Knowledge Graph Pipeline Summary\n\n"
            f"## Video Information\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Video ID** | `{final_result.related_video_id}` |\n"
            f"| **Model Used** | `{final_result.llm_model or 'N/A'}` |\n\n"
            f"## Entity Statistics\n"
            f"| Field | Count |\n"
            f"|-------|-------|\n"
            f"| **Raw Entities (extracted)** | `{final_result.total_raw_entities}` |\n"
            f"| **Canonical Entities (resolved)** | `{final_result.total_canonical_entities}` |\n"
            f"| **Entity Resolution Ratio** | `{final_result.total_raw_entities / final_result.total_canonical_entities:.2f}x` |\n"
            f"| **Global Relationships** | `{final_result.total_relationships}` |\n\n"
            f"## Event Layer\n"
            f"| Field | Count |\n"
            f"|-------|-------|\n"
            f"| **Big Events** | `{final_result.total_events}` |\n"
            f"| **Micro-Events** | `{final_result.total_micro_events}` |\n"
            f"| **Event-to-Event Edges** | `{final_result.total_event_edges}` |\n"
            f"| **Micro-Event Edges** | `{final_result.total_micro_event_edges}` |\n\n"
            f"## Community Structure\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Communities Detected** | `{final_result.total_communities}` |\n"
            f"| **Membership Edges** | `{final_result.total_membership_edges}` |\n"
            f"| **Graph Modularity** | `{final_result.graph_modularity:.4f}` |\n\n"
            f"## Node2Vec Embeddings\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Nodes with Embeddings** | `{final_result.total_nodes_with_embeddings}` |\n"
            f"| **Embedding Dimension** | `{final_result.node2vec_meta.get('dim', 'N/A')}` |\n\n"
            f"## Cost & Usage\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **LLM Calls** | `{final_result.llm_calls}` |\n"
            f"| **Total Tokens (Prompt)** | `{tokens_str}` |\n"
            f"| **Estimated Cost** | `{cost_str}` |\n"
        )

        await acreate_markdown_artifact(
            key=f"kg-pipeline-{final_result.related_video_id}".lower(),
            markdown=markdown,
            description=f"KG pipeline summary for video {final_result.related_video_id}",
        )

@task(**{**KG_PIPELINE_CONFIG.to_task_kwargs(), "name": "KG Pipeline"}) #type:ignore
async def kg_pipeline_task(
    segments: list[SegmentCaptionArtifact],
) -> KGGraphArtifact:
    """Run the complete Knowledge Graph pipeline.

    Args:
        segments: List of SegmentCaptionArtifact

    Returns:
        KGGraphArtifact with all KG pipeline outputs
    """
    logger = get_run_logger()
    settings = get_settings()
    kwargs = KG_PIPELINE_CONFIG.additional_kwargs

    logger.info(f"[KGPipeline] Starting | {len(segments)} segment(s)")

    minio_client = MinioStorageClient(
        endpoint=settings.minio.endpoint,
        access_key=settings.minio.access_key,
        secret_key=settings.minio.secret_key,
        secure=settings.minio.secure,
    )
    postgres_client = await get_postgres_client()

    model = kwargs.get("model", "qwen/qwen3-coder-next")
    base_url = kwargs.get("base_url", "https://openrouter.ai/api/v1")
    max_tokens = kwargs.get("max_tokens", 8192)  #
    llm_config = OpenRouterConfig(
        model=model,
        base_url=base_url,
        max_tokens=max_tokens,
        api_key=SecretStr(os.environ.get("OPENROUTER_API_KEY", "")),
    )
    llm_client = OpenRouterClient(config=llm_config)

    dense_base_url = kwargs.get("dense_embedding_base_url", "http://mmbert:8000")
    dense_client = MMBertClient(MMBertConfig(base_url=dense_base_url))

    sparse_url = kwargs.get("sparse_embedding_url", "triton:8001")
    sparse_client = SpladeClient(SpladeConfig(url=sparse_url))

    user_id = segments[0].user_id if segments else "unknown"
    segment_caption_artifact_ids = [s.artifact_id for s in segments]

    task_impl = KGPipelineTask(
        artifact_visitor=ArtifactPersistentVisitor(minio_client, postgres_client),
        minio_client=minio_client,
        user_id=user_id,
        segment_caption_artifact_ids=segment_caption_artifact_ids,
    )

    clients = {
        "llm": llm_client,
        "dense": dense_client,
        "sparse": sparse_client,
    }

    try:
        results = await task_impl.execute_template(segments, clients)
    finally:
        await llm_client.close()
        await dense_client.close()
        await shutdown_postgres_client(postgres_client)

    logger.info(f"[KGPipeline] Done | KG artifact {results.artifact_id}")
    return results