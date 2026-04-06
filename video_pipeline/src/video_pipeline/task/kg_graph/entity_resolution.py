"""Entity Resolution - Stage 2.

Resolve duplicate entities across segments using hybrid embeddings and LLM.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict

import numpy as np
import llm_json
from pydantic import BaseModel, Field, field_validator
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from langchain_core.messages import HumanMessage

from video_pipeline.core.client.inference import MMBertClient, SpladeClient

from .models import (
    KGSegment,
    CanonicalEntity,
    GlobalRelationship,
    SegmentView,
    ResolvedKG,
    CostTracker,
)


class ResolvedSubGroup(BaseModel):
    local_ids: list[str] = Field(
        ...,
        description="The local_ids of entities that refer to the SAME real-world object/person.",
    )
    merged_desc: str | None = Field(
        default=None,
        description="A single, unified description synthesised from all members.",
    )
    canonical_name: str | None = Field(
        default=None,
        description="The single best name to use for this entity going forward. If omitted, will be extracted from merged_desc.",
    )


class EntityOutput(BaseModel):
    entity_groups: list[ResolvedSubGroup] = Field(
        ...,
        description="List of resolved sub-groups, one per distinct real-world entity.",
    )



def load_and_flatten(kg_segments: list[KGSegment]) -> list[dict]:
    """Flatten entities from KGSegments with segment context."""
    kg_segments.sort(key=lambda x: x.start_time)
    entities: list[dict] = []

    for seg_idx, segment in enumerate(kg_segments):
        for ent in segment.entities:
            ent_dict = ent.model_dump()
            ent_dict["belong_index"] = seg_idx
            ent_dict["local_id"] = f"loc_{uuid.uuid4().hex[:10]}"
            ent_dict["text_rep"] = f"{ent.entity_name}: {ent.desc}"
            entities.append(ent_dict)

    return entities


async def build_hybrid_clusters(
    entities: list[dict],
    dense_client: MMBertClient,
    sparse_client: SpladeClient,
    dense_weight: float = 0.9,
    sparse_weight: float = 0.1,
    sim_threshold: float = 0.75,
) -> dict[str, list[dict]]:
    """Build hybrid clusters using dense + sparse embeddings from inference clients."""
    texts = [e["text_rep"] for e in entities]
    print(f"  Embedding {len(texts)} entities...")

    dense_vecs = await dense_client.ainfer(texts)
    if dense_vecs is None:
        raise RuntimeError("Failed to get dense embeddings from MMBertClient")
    dense_vecs = np.array(dense_vecs)
    norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    dense_vecs = dense_vecs / norms
    dense_sim = cosine_similarity(dense_vecs)

    for idx, e in enumerate(entities):
        e["_dense_vec"] = dense_vecs[idx].tolist()

    sparse_vectors = await sparse_client.aencode(texts)

    max_dim = max(
        (max(sv.indices) if sv.indices else 0 for sv in sparse_vectors),
        default=1,
    ) + 1

    rows, cols, vals = [], [], []
    for i, sv in enumerate(sparse_vectors):
        if sv.indices:
            rows.extend([i] * len(sv.indices))
            cols.extend(sv.indices)
            vals.extend(sv.values)

    sparse_mat = csr_matrix((vals, (rows, cols)), shape=(len(texts), max_dim))
    sparse_sim = cosine_similarity(sparse_mat)

    hybrid_sim = dense_weight * dense_sim + sparse_weight * sparse_sim
    hybrid_dist = np.clip(1.0 - hybrid_sim, 0.0, 2.0)

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - sim_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = model.fit_predict(hybrid_dist)

    clusters: dict[str, list[dict]] = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[str(label)].append(entities[idx])

    print(f"  → {len(clusters)} candidate clusters from {len(entities)} entities.")
    return dict(clusters)


async def resolve_cluster(
    cluster_id: str,
    entities_in_cluster: list[dict],
    structured_llm,
    semaphore: asyncio.Semaphore,
    cost_tracker: CostTracker,
) -> list[dict]:
    """Resolve a single cluster using LLM verification."""
    if len(entities_in_cluster) == 1:
        e = entities_in_cluster[0]
        e["global_entity_id"] = f"GLOBAL_{uuid.uuid4().hex[:8]}"
        e["merged_desc"] = e.get("desc", "")
        e["canonical_name"] = e.get("entity_name", "")
        return entities_in_cluster

    entity_context = [
        {
            "local_id": e["local_id"],
            "name": e.get("entity_name", ""),
            "desc": e.get("desc", ""),
            "time_index": e.get("belong_index", -1),
        }
        for e in entities_in_cluster
    ]

    prompt = f"""You are an Entity Resolution expert working on a knowledge graph extracted from a continuous video.

I will give you a cluster of entities that an embedding model grouped together as potential duplicates.

Your tasks:
1. GROUP  — Decide which entities refer to the SAME real-world object/person.
2. MERGE  — For each group, write ONE unified description.
3. NAME   — Pick the single best canonical name for each group.

Rules:
- Different colours, sizes, or roles = different entities → separate groups.
- Every local_id in the input MUST appear in exactly one group.
- merged_desc: present tense, 2-4 sentences, no repetition.
- canonical_name: prefer the most descriptive / specific name.

Input entities:
{entity_context}


Each entity_group represents one real-world entity with its merged local_ids, unified description, and canonical name.
"""

    msg = HumanMessage(content=prompt)

    async with semaphore:
        try:
            parsed, usage = await structured_llm([msg])
            prompt_tokens = usage.get('prompt_tokens', 0) or 0
            completion_tokens = usage.get('completion_tokens', 0) or 0
            cost = usage.get('cost', 0.0) or 0.0
            cost_tracker.add_usage(prompt_tokens, completion_tokens, cost)

            all_local_ids = {e["local_id"] for e in entities_in_cluster}
            seen = set()
            resolved: list[dict] = []

            for sub_group in parsed.entity_groups:
                global_id = f"GLOBAL_{uuid.uuid4().hex[:8]}"
                merged_desc = sub_group.merged_desc
                canonical_name = sub_group.canonical_name
                
                if not merged_desc:
                    descs = []
                    for local_id in sub_group.local_ids:
                        for e in entities_in_cluster:
                            if e["local_id"] == local_id and e.get("desc"):
                                descs.append(e["desc"])
                                break
                    merged_desc = " ".join(descs) if descs else ""
                
                if not canonical_name:
                    first_sentence = merged_desc.split('.')[0].strip()
                    canonical_name = first_sentence[:50] if len(first_sentence) > 50 else first_sentence
                
                

                for local_id in sub_group.local_ids:
                    if local_id not in all_local_ids or local_id in seen:
                        continue
                    seen.add(local_id)
                    for e in entities_in_cluster:
                        if e["local_id"] == local_id:
                            e["global_entity_id"] = global_id
                            e["merged_desc"] = merged_desc
                            e["canonical_name"] = canonical_name
                            resolved.append(e)
                            break

            for e in entities_in_cluster:
                if "global_entity_id" not in e:
                    e["global_entity_id"] = f"GLOBAL_{uuid.uuid4().hex[:8]}"
                    e["merged_desc"] = e.get("desc", "")
                    e["canonical_name"] = e.get("entity_name", "")
                    resolved.append(e)

            return resolved

        except Exception as exc:
            print(f"  [WARN] cluster {cluster_id} LLM error: {exc}")
            for e in entities_in_cluster:
                e["global_entity_id"] = f"UNRESOLVED_{uuid.uuid4().hex[:8]}"
                e["merged_desc"] = e.get("desc", "")
                e["canonical_name"] = e.get("entity_name", "")
            return entities_in_cluster


async def run_llm_resolution(
    clusters: dict[str, list[dict]],
    llm_client,
    max_concurrent: int = 5,
    cost_tracker: CostTracker | None = None,
) -> list[dict]:
    """Run LLM resolution on all clusters."""
    structured_llm = llm_client.as_structured_llm(EntityOutput)
    semaphore = asyncio.Semaphore(max_concurrent)

    if cost_tracker is None:
        cost_tracker = CostTracker()

    tasks = [
        resolve_cluster(cid, ents, structured_llm, semaphore, cost_tracker)
        for cid, ents in clusters.items()
    ]

    all_resolved: list[dict] = []
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="  LLM verification"):
        result = await future
        all_resolved.extend(result)

    return all_resolved


async def build_canonical_entities(
    resolved: list[dict],
    dense_client: MMBertClient,
) -> dict[str, CanonicalEntity]:
    """Build canonical entity table from resolved entities."""
    canonical: dict[str, CanonicalEntity] = {}

    for e in resolved:
        gid = e["global_entity_id"]
        if gid not in canonical:
            canonical[gid] = CanonicalEntity(
                video_id=e.get("video_id", ""),
                global_entity_id=gid,
                entity_name=e.get("canonical_name") or e.get("entity_name", ""),
                entity_type=e.get("entity_type", ""),
                desc=e.get("merged_desc") or e.get("desc", ""),
            )
        canonical[gid].merged_from.append(e.get("entity_id", ""))

    gids = list(canonical.keys())
    descs = [canonical[gid].desc for gid in gids]
    vecs = await dense_client.ainfer(descs)
    if vecs is None:
        raise RuntimeError("Failed to embed canonical entity descriptions")
    vecs = np.array(vecs)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    vecs = vecs / norms
    for gid, vec in zip(gids, vecs):
        canonical[gid].semantic_embedding = vec.tolist()

    return canonical


def remap_relationships(
    kg_segments: list[KGSegment],
    orig_id_to_global: dict[str, str],
    video_id: str = "",
) -> list[GlobalRelationship]:
    """Remap relationships to global entity IDs and collapse duplicates."""
    triple_registry: dict[tuple, GlobalRelationship] = {}

    for seg_idx, segment in enumerate(kg_segments):
        for rel in segment.relationships:
            subj_global = orig_id_to_global.get(rel.subject_id, rel.subject_id)
            obj_global = orig_id_to_global.get(rel.object_id, rel.object_id)

            if subj_global == obj_global:
                continue

            key = (subj_global, rel.relation_desc, obj_global)
            if key not in triple_registry:
                triple_registry[key] = GlobalRelationship(
                    video_id=video_id,
                    subject_global=subj_global,
                    relation_desc=rel.relation_desc,
                    object_global=obj_global,
                    weight=0,
                )
            triple_registry[key].weight += 1
            triple_registry[key].seen_in_segments.append(seg_idx)

    return list(triple_registry.values())


def build_resolved_kg(
    kg_segments: list[KGSegment],
    resolved_entities: list[dict],
    canonical: dict[str, CanonicalEntity],
    video_id: str = "",
) -> ResolvedKG:
    """Build the final resolved KG."""
    orig_to_global: dict[str, str] = {
        e["entity_id"]: e["global_entity_id"]
        for e in resolved_entities
        if "entity_id" in e and "global_entity_id" in e
    }

    global_to_segments: dict[str, set] = defaultdict(set)
    for e in resolved_entities:
        if "global_entity_id" in e and "belong_index" in e:
            global_to_segments[e["global_entity_id"]].add(e["belong_index"])

    for gid, rec in canonical.items():
        idxs = sorted(global_to_segments.get(gid, set()))
        rec.first_seen_segment = idxs[0] if idxs else None
        rec.last_seen_segment = idxs[-1] if idxs else None

    global_relationships = remap_relationships(kg_segments, orig_to_global, video_id)

    kg_segments.sort(key=lambda x: x.start_time)
    segments_out: list[SegmentView] = []
    for seg_idx, segment in enumerate(kg_segments):
        seg_global_ids = {
            orig_to_global.get(ent.entity_id, "")
            for ent in segment.entities
        } - {""}

        seg_entities = [canonical[gid] for gid in seg_global_ids if gid in canonical]

        seg_rels = [
            r for r in global_relationships
            if r.subject_global in seg_global_ids
            and r.object_global in seg_global_ids
        ]

        segments_out.append(SegmentView(
            video_id=video_id,
            segment_index=seg_idx,
            from_batch=segment.from_batch,
            to_batch=segment.to_batch,
            start_time=segment.start_time,
            end_time=segment.end_time,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            caption=segment.summary_caption,
            entities=seg_entities,
            relationships=seg_rels,
            events=list(segment.events),
        ))

    return ResolvedKG(
        video_id=video_id,
        entities=list(canonical.values()),
        relationships=global_relationships,
        segments=segments_out,
    )


async def run_entity_resolution(
    kg_segments: list[KGSegment],
    llm_client,
    dense_client: MMBertClient,
    sparse_client: SpladeClient,
    video_id: str,
    dense_weight: float = 0.9,
    sparse_weight: float = 0.1,
    sim_threshold: float = 0.75,
    max_concurrent: int = 5,
    cost_tracker: CostTracker | None = None,
) -> ResolvedKG:
    """Run the full entity resolution pipeline."""
    if cost_tracker is None:
        cost_tracker = CostTracker()

    print(f"\n[Stage 2] Entity Resolution")
    print(f"  Flattening entities from {len(kg_segments)} segments...")

    flat_entities = load_and_flatten(kg_segments)
    print(f"  {len(flat_entities)} raw entities across all segments.")

    clusters = await build_hybrid_clusters(
        flat_entities,
        dense_client,
        sparse_client,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        sim_threshold=sim_threshold,
    )

    print(f"  Running LLM resolution on {len(clusters)} clusters...")
    resolved_entities = await run_llm_resolution(clusters, llm_client, max_concurrent, cost_tracker)

    n_global = len({e["global_entity_id"] for e in resolved_entities})
    print(f"  {len(flat_entities)} raw → {n_global} canonical entities.")

    canonical = await build_canonical_entities(resolved_entities, dense_client)
    resolved_kg = build_resolved_kg(kg_segments, resolved_entities, canonical, video_id)

    print(f"  {len(resolved_kg.relationships)} unique global relationships.")

    return resolved_kg