"""ArangoDB Indexing Task.

Inserts Knowledge Graph data into ArangoDB for graph-based retrieval.
This task takes the KGGraphArtifact output from the KG pipeline and loads:
- Vertex collections: videos, entities, events, micro_events, communities
- Edge collections: entity_relations, event_sequences, event_entities,
  micro_event_sequences, micro_event_parents, micro_event_entities,
  community_members, event_communities

Note: Vector indexes are NOT created here - they should be created separately
after data is loaded (ArangoDB IVF indexes need training data).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from prefect import get_run_logger, task
from prefect.artifacts import acreate_markdown_artifact

from video_pipeline.task.base.base_task import TaskConfig, BaseTask
from video_pipeline.core.client.progress import StageRegistry
from video_pipeline.core.artifact import KGGraphArtifact, ArangoIndexingArtifact
from video_pipeline.core.client.storage.arango import ArangoStorageClient, ArangoConfig
from video_pipeline.core.client.inference import MMBertClient, MMBertConfig
from video_pipeline.config import get_settings


ARANGO_INDEXING_CONFIG = TaskConfig.from_yaml("arango_indexing")


def _ns(video_id: str, local_key: str) -> str:
    """Namespace a local key with video_id."""
    return f"{video_id}::{local_key}"


def _strip_collection(arango_id: str) -> str:
    """Strip collection prefix from ArangoDB ID.

    events/event_0001 -> event_0001

    Raises:
        ValueError: If arango_id is empty or doesn't contain '/'
    """
    if not arango_id:
        raise ValueError("arango_id is empty - check that the edge data has the correct key (_from/_to)")
    if "/" not in arango_id:
        raise ValueError(f"arango_id '{arango_id}' is missing collection prefix (expected format: collection/key)")
    return arango_id.split("/")[1]


@StageRegistry.register
class ArangoIndexingTask(BaseTask[KGGraphArtifact, ArangoIndexingArtifact]):
    """Index KG data into ArangoDB."""

    config = ARANGO_INDEXING_CONFIG

    async def preprocess(
        self,
        input_data: KGGraphArtifact,
    ) -> KGGraphArtifact:
        """Validate input artifact."""
        logger = get_run_logger()
        logger.info(f"[ArangoIndexing] Preprocessing KG artifact {input_data.artifact_id}")
        return input_data

    async def execute(
        self,
        preprocessed: KGGraphArtifact,
        client: dict[str, Any],
    ) -> ArangoIndexingArtifact:
        """Insert KG data into ArangoDB."""
        logger = get_run_logger()
        arango_client: ArangoStorageClient = client["arango"]
        dense_client: MMBertClient = client["dense"]

        video_id = preprocessed.related_video_id
        stats: dict[str, int] = {}

        db = arango_client.setup_database()

        db.collection("videos").insert(
            {
                "_key": video_id,
                "video_id": video_id,
                "entity_count": len(preprocessed.entities),
                "event_count": len(preprocessed.events),
                "micro_event_count": len(preprocessed.micro_event_nodes),
                "community_count": len(preprocessed.communities),
                "segment_count": len(preprocessed.segment_views),
                "created_at": datetime.now().isoformat(),
            },
            overwrite_mode="replace",
        )
        logger.info(f"[ArangoIndexing] Inserted video document: {video_id}")

        nodes = preprocessed.node_embeddings

        entity_docs = []
        entity_texts = []

        for entity in preprocessed.entities:
            gid = entity.get("global_entity_id", "")
            text = f"{entity.get('entity_name', '')}. {entity.get('desc', '')}"
            entity_texts.append(text)

        if entity_texts:
            entity_embeddings = await dense_client.ainfer(entity_texts) or []

            for idx, entity in enumerate(preprocessed.entities):
                gid = entity.get("global_entity_id", "")
                n2v_node = nodes.get(gid, {})
                sem_emb = entity_embeddings[idx] if idx < len(entity_embeddings) else []

                entity_docs.append({
                    "_key": _ns(video_id, gid),
                    "video_id": video_id,
                    "global_entity_id": gid,
                    "entity_name": entity.get("entity_name", ""),
                    "entity_type": entity.get("entity_type", ""),
                    "desc": entity.get("desc", ""),
                    "merged_from": entity.get("merged_from", []),
                    "first_seen_segment": entity.get("first_seen_segment"),
                    "last_seen_segment": entity.get("last_seen_segment"),
                    "semantic_embedding": sem_emb,
                    "structural_embedding_entity_only": n2v_node.get("entity_only_embedding"),
                    "structural_embedding_entity_event": n2v_node.get("entity_event_embedding"),
                    "structural_embedding_full": n2v_node.get("full_heterogeneous_embedding"),
                })

        if entity_docs:
            db.collection("entities").insert_many(entity_docs, overwrite_mode="replace")
            stats["entities"] = len(entity_docs)
            logger.info(f"[ArangoIndexing] Inserted {len(entity_docs)} entities")

        event_docs = []
        event_texts = [e.get("caption", "") for e in preprocessed.events]

        if event_texts:
            event_embeddings = await dense_client.ainfer(event_texts) or []

            for idx, event in enumerate(preprocessed.events):
                ekey = event.get("_key", event.get("key", ""))
                n2v_node = nodes.get(ekey, {})
                sem_emb = event_embeddings[idx] if idx < len(event_embeddings) else []

                event_docs.append({
                    "_key": _ns(video_id, ekey),
                    "video_id": video_id,
                    "segment_index": event.get("segment_index"),
                    "start_time": event.get("start_time"),
                    "end_time": event.get("end_time"),
                    "start_sec": event.get("start_sec"),
                    "end_sec": event.get("end_sec"),
                    "caption": event.get("caption", ""),
                    "micro_events": event.get("micro_events", []),
                    "entities_global": event.get("entities_global", []),
                    "semantic_embedding": sem_emb,
                    "structural_embedding_entity_only": n2v_node.get("entity_only_embedding"),
                    "structural_embedding_entity_event": n2v_node.get("entity_event_embedding"),
                    "structural_embedding_full": n2v_node.get("full_heterogeneous_embedding"),
                })

        if event_docs:
            db.collection("events").insert_many(event_docs, overwrite_mode="replace")
            stats["events"] = len(event_docs)
            logger.info(f"[ArangoIndexing] Inserted {len(event_docs)} events")

        micro_docs = []
        micro_texts = [m.get("text", "") for m in preprocessed.micro_event_nodes]

        if micro_texts:
            micro_embeddings = await dense_client.ainfer(micro_texts) or []

            for idx, mn in enumerate(preprocessed.micro_event_nodes):
                mkey = mn.get("_key", mn.get("key", ""))
                n2v_node = nodes.get(mkey, {})
                sem_emb = micro_embeddings[idx] if idx < len(micro_embeddings) else []

                micro_docs.append({
                    "_key": _ns(video_id, mkey),
                    "video_id": video_id,
                    "parent_event_key": _strip_collection(mn.get("parent_event_key", "")),
                    "segment_index": mn.get("segment_index"),
                    "micro_index": mn.get("micro_index"),
                    "start_time": mn.get("start_time"),
                    "end_time": mn.get("end_time"),
                    "start_secs": mn.get("start_secs"),
                    "end_secs": mn.get("end_secs"),
                    "text": mn.get("text", ""),
                    "related_caption_context": mn.get("related_caption_context", ""),
                    "entities_global": mn.get("entities_global", []),
                    "semantic_embedding": sem_emb,
                    "structural_embedding_entity_event": n2v_node.get("entity_event_embedding"),
                    "structural_embedding_full": n2v_node.get("full_heterogeneous_embedding"),
                })

        if micro_docs:
            db.collection("micro_events").insert_many(micro_docs, overwrite_mode="replace")
            stats["micro_events"] = len(micro_docs)
            logger.info(f"[ArangoIndexing] Inserted {len(micro_docs)} micro-events")

        comm_docs = []
        comm_texts = []
        for comm in preprocessed.communities:
            if not comm.get("embedding"):
                text = f"{comm.get('title', '')}. {comm.get('summary', '')}"
                comm_texts.append(text)

        if comm_texts:
            comm_embeddings = await dense_client.ainfer(comm_texts) or []
            emb_idx = 0

            for comm in preprocessed.communities:
                ckey = comm.get("comm_key", "")
                n2v_node = nodes.get(ckey, {})
                sem_emb = comm.get("embedding")

                if not sem_emb:
                    sem_emb = comm_embeddings[emb_idx] if emb_idx < len(comm_embeddings) else []
                    emb_idx += 1

                comm_docs.append({
                    "_key": _ns(video_id, ckey),
                    "video_id": video_id,
                    "comm_idx": comm.get("comm_idx"),
                    "title": comm.get("title", ""),
                    "summary": comm.get("summary", ""),
                    "size": comm.get("size", 0),
                    "member_keys": comm.get("member_keys", []),
                    "event_keys": comm.get("event_keys", []),
                    "semantic_embedding": sem_emb,
                    "structural_embedding_full": n2v_node.get("full_heterogeneous_embedding"),
                })

        if comm_docs:
            db.collection("communities").insert_many(comm_docs, overwrite_mode="replace")
            stats["communities"] = len(comm_docs)
            logger.info(f"[ArangoIndexing] Inserted {len(comm_docs)} communities")

        # === Insert Entity Relations ===
        rel_docs = []
        for rel in preprocessed.relationships:
            subj = rel.get("subject_global", "")
            obj = rel.get("object_global", "")
            if not subj or not obj:
                continue

            rel_docs.append({
                "_from": f"entities/{_ns(video_id, subj)}",
                "_to": f"entities/{_ns(video_id, obj)}",
                "video_id": video_id,
                "relation_type": rel.get("relation_desc", ""),
                "weight": rel.get("weight", 1),
                "seen_in_segments": rel.get("seen_in_segments", []),
            })

        if rel_docs:
            db.collection("entity_relations").insert_many(rel_docs, overwrite_mode="replace")
            stats["entity_relations"] = len(rel_docs)
            logger.info(f"[ArangoIndexing] Inserted {len(rel_docs)} entity relations")

        # === Insert Event-Entity Links ===
        ee_docs = []
        for link in preprocessed.event_entity_links:
            fk = _strip_collection(link.get("_from", ""))
            tk = _strip_collection(link.get("_to", ""))
            if not fk or not tk:
                continue

            ee_docs.append({
                "_from": f"events/{_ns(video_id, fk)}",
                "_to": f"entities/{_ns(video_id, tk)}",
                "video_id": video_id,
            })

        if ee_docs:
            db.collection("event_entities").insert_many(ee_docs, overwrite_mode="replace")
            stats["event_entities"] = len(ee_docs)
            logger.info(f"[ArangoIndexing] Inserted {len(ee_docs)} event-entity links")

        # === Insert Event Sequences ===
        es_docs = []
        for edge in preprocessed.event_edges:
            from_key = _strip_collection(edge.get("_from", ""))
            to_key = _strip_collection(edge.get("_to", ""))
            if not from_key or not to_key:
                continue

            es_docs.append({
                "_from": f"events/{_ns(video_id, from_key)}",
                "_to": f"events/{_ns(video_id, to_key)}",
                "video_id": video_id,
                "edge_type": edge.get("edge_type", ""),
                "similarity": edge.get("similarity"),
                "temporal_gap_s": edge.get("temporal_gap_s"),
                "shared_entities": edge.get("shared_entities", 0),
                "llm_reason": edge.get("llm_reason"),
            })

        if es_docs:
            db.collection("event_sequences").insert_many(es_docs, overwrite_mode="replace")
            stats["event_sequences"] = len(es_docs)
            logger.info(f"[ArangoIndexing] Inserted {len(es_docs)} event sequences")

        me_docs = []
        for edge in preprocessed.micro_event_edges:
            from_key = _strip_collection(edge.get("_from", ""))
            to_key = _strip_collection(edge.get("_to", ""))
            if not from_key or not to_key:
                continue

            me_docs.append({
                "_from": f"micro_events/{_ns(video_id, from_key)}",
                "_to": f"micro_events/{_ns(video_id, to_key)}",
                "video_id": video_id,
                "edge_type": edge.get("edge_type", ""),
                "similarity": edge.get("similarity"),
                "temporal_gap_s": edge.get("temporal_gap_s"),
                "shared_entities": edge.get("shared_entities", 0),
                "llm_reason": edge.get("llm_reason"),
            })

        if me_docs:
            db.collection("micro_event_sequences").insert_many(me_docs, overwrite_mode="replace")
            stats["micro_event_sequences"] = len(me_docs)
            logger.info(f"[ArangoIndexing] Inserted {len(me_docs)} micro-event sequences")

        mp_docs = []
        for mn in preprocessed.micro_event_nodes:
            mkey = mn.get("_key", mn.get("key", ""))
            parent = _strip_collection(mn.get("parent_event_key", ""))
            if not mkey or not parent:
                continue

            mp_docs.append({
                "_from": f"micro_events/{_ns(video_id, mkey)}",
                "_to": f"events/{_ns(video_id, parent)}",
                "video_id": video_id,
            })

        if mp_docs:
            db.collection("micro_event_parents").insert_many(mp_docs, overwrite_mode="replace")
            stats["micro_event_parents"] = len(mp_docs)
            logger.info(f"[ArangoIndexing] Inserted {len(mp_docs)} micro-event parents")

        mee_docs = []
        for mn in preprocessed.micro_event_nodes:
            mkey = mn.get("_key", mn.get("key", ""))
            for gid in mn.get("entities_global", []):
                mee_docs.append({
                    "_from": f"micro_events/{_ns(video_id, mkey)}",
                    "_to": f"entities/{_ns(video_id, gid)}",
                    "video_id": video_id,
                })

        if mee_docs:
            db.collection("micro_event_entities").insert_many(mee_docs, overwrite_mode="replace")
            stats["micro_event_entities"] = len(mee_docs)
            logger.info(f"[ArangoIndexing] Inserted {len(mee_docs)} micro-event entities")

        cm_docs = []
        for edge in preprocessed.membership_edges:
            from_key = _strip_collection(edge.get("_from", ""))
            to_key = _strip_collection(edge.get("_to", ""))
            if not from_key or not to_key:
                continue

            cm_docs.append({
                "_from": f"entities/{_ns(video_id, from_key)}",
                "_to": f"communities/{_ns(video_id, to_key)}",
                "video_id": video_id,
            })

        if cm_docs:
            db.collection("community_members").insert_many(cm_docs, overwrite_mode="replace")
            stats["community_members"] = len(cm_docs)
            logger.info(f"[ArangoIndexing] Inserted {len(cm_docs)} community members")

        ec_docs = []
        for edge in preprocessed.event_community_edges:
            from_key = _strip_collection(edge.get("_from", ""))
            to_key = _strip_collection(edge.get("_to", ""))
            if not from_key or not to_key:
                continue

            ec_docs.append({
                "_from": f"events/{_ns(video_id, from_key)}",
                "_to": f"communities/{_ns(video_id, to_key)}",
                "video_id": video_id,
                "shared_entities": edge.get("shared_entities", 0),
                "assignment": edge.get("assignment", ""),
            })

        if ec_docs:
            db.collection("event_communities").insert_many(ec_docs, overwrite_mode="replace")
            stats["event_communities"] = len(ec_docs)
            logger.info(f"[ArangoIndexing] Inserted {len(ec_docs)} event-community edges")

        artifact = ArangoIndexingArtifact(
            user_id=preprocessed.user_id,
            related_video_id=video_id,
            related_kg_artifact_id=preprocessed.artifact_id,
            **stats, #type: ignore
        )

        logger.info(f"[ArangoIndexing] Complete | Total docs: {sum(stats.values())}")

        return artifact

    async def postprocess(
        self,
        result: ArangoIndexingArtifact,
    ) -> ArangoIndexingArtifact:
        """Persist the artifact to PostgreSQL."""
        logger = get_run_logger()

        if self.artifact_visitor:
            await self.artifact_visitor.visit_artifact(result)
            logger.info(f"[ArangoIndexing] Persisted artifact {result.artifact_id}")

        return result

    @staticmethod
    async def summary_artifact(final_result: ArangoIndexingArtifact) -> None:
        """Create a Prefect artifact summary."""
        if not final_result.entities:
            return

        markdown = (
            f"# ArangoDB Indexing Summary\n\n"
            f"## Video Information\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| **Video ID** | `{final_result.related_video_id}` |\n"
            f"| **KG Artifact ID** | `{final_result.related_kg_artifact_id}` |\n\n"
            f"## Vertex Collections\n"
            f"| Collection | Count |\n"
            f"|------------|-------|\n"
            f"| **Entities** | `{final_result.entities}` |\n"
            f"| **Events** | `{final_result.events}` |\n"
            f"| **Micro-Events** | `{final_result.micro_events}` |\n"
            f"| **Communities** | `{final_result.communities}` |\n\n"
            f"## Edge Collections\n"
            f"| Collection | Count |\n"
            f"|------------|-------|\n"
            f"| **Entity Relations** | `{final_result.entity_relations}` |\n"
            f"| **Event Sequences** | `{final_result.event_sequences}` |\n"
            f"| **Event Entities** | `{final_result.event_entities}` |\n"
            f"| **Micro-Event Sequences** | `{final_result.micro_event_sequences}` |\n"
            f"| **Micro-Event Parents** | `{final_result.micro_event_parents}` |\n"
            f"| **Micro-Event Entities** | `{final_result.micro_event_entities}` |\n"
            f"| **Community Members** | `{final_result.community_members}` |\n"
            f"| **Event Communities** | `{final_result.event_communities}` |\n"
        )

        await acreate_markdown_artifact(
            key=f"arango-indexing-{final_result.related_video_id}".lower(),
            markdown=markdown,
            description=f"ArangoDB indexing summary for video {final_result.related_video_id}",
        )


@task(**{**ARANGO_INDEXING_CONFIG.to_task_kwargs(), "name": "ArangoDB Indexing"})  # type: ignore
async def arango_indexing_task(
    kg_artifact: KGGraphArtifact,
) -> ArangoIndexingArtifact:
    """Index KG data into ArangoDB.

    Args:
        kg_artifact: KGGraphArtifact from the KG pipeline

    Returns:
        ArangoIndexingArtifact with insertion stats

    Raises:
        RuntimeError: If any ArangoDB operation fails
    """
    logger = get_run_logger()
    settings = get_settings()

    logger.info(f"[ArangoIndexing] Starting | Video: {kg_artifact.related_video_id}")

    arango_config = ArangoConfig(
        host=settings.arango.host,
        database=settings.arango.database,
        graph_name=settings.arango.graph_name,
    )
    arango_client = ArangoStorageClient(config=arango_config)

    kwargs = ARANGO_INDEXING_CONFIG.additional_kwargs
    dense_base_url = kwargs.get("dense_embedding_base_url", "http://mmbert:8000")
    dense_client = MMBertClient(MMBertConfig(base_url=dense_base_url))

    user_id = kg_artifact.user_id

    try:
        arango_client.connect()

        task_impl = ArangoIndexingTask(
            artifact_visitor=None,  #type:ignore
            minio_client=None,   #type:ignore
            user_id=user_id,
        )

        clients = {
            "arango": arango_client,
            "dense": dense_client,
        }

        result = await task_impl.execute_template(kg_artifact, clients)
    finally:
        arango_client.disconnect()
        await dense_client.close()

    logger.info(f"[ArangoIndexing] Done | Artifact: {result.artifact_id}")
    return result