"""Event Linking - Stage 3.

Build event/micro-event nodes and edges with semantic, Jaccard, and LLM linking.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import cast

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from llama_index.core.llms import ChatMessage, TextBlock

from video_pipeline.core.client.inference import MMBertClient

from .models import (
    ResolvedKG,
    EventNode,
    EventEntityEdge,
    EventEdge,
    MicroEventNode,
    MicroEventEdge,
    EnhancedKG,
    CostTracker,
)


# LLM-bound models
class EventLinkOutput(BaseModel):
    should_link: bool = Field(..., description="True if the two events are meaningfully related.")
    link_type: str = Field(..., description="One of: SEMANTICALLY_SIMILAR, SHARES_CONTEXT, CAUSAL, or UNRELATED.")
    reason: str = Field(..., description="One-sentence justification.")


class MicroLinkOutput(BaseModel):
    should_link: bool = Field(..., description="True if the two micro-events are meaningfully related.")
    link_type: str = Field(
        ...,
        description="One of: SEMANTICALLY_SIMILAR_MICRO, SHARES_CONTEXT_MICRO, CAUSAL_MICRO, or UNRELATED.",
    )
    reason: str = Field(..., description="One-sentence justification.")


# Helpers
def format_audio_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:04.1f}"


def _time_gap(start_sec: float, end_sec: float) -> str:
    gap = max(0.0, end_sec - start_sec)
    return format_audio_time(gap)


# Stage 3 — Event layer

async def build_event_docs(
    resolved_kg: ResolvedKG,
    dense_client: MMBertClient,
) -> list[EventNode]:
    """Create one EventNode per segment, embed captions."""
    event_nodes: list[EventNode] = []

    for seg in resolved_kg.segments:
        seg_idx = seg.segment_index

        segment_global_entities = list({
            e.global_entity_id for e in seg.entities
        })

        event_nodes.append(EventNode(
            video_id=resolved_kg.video_id,
            key=f"event_{seg_idx:04d}",
            segment_index=seg_idx,
            start_time=seg.start_time,
            end_time=seg.end_time,
            start_sec=seg.start_sec,
            end_sec=seg.end_sec,
            caption=seg.caption,
            entities_global=segment_global_entities,
            micro_events=[ev.event_des for ev in seg.events],
            embedding=None,
        ))

    print(f"  Embedding {len(event_nodes)} event captions...")
    captions = [e.caption for e in event_nodes]
    dense_vecs = await dense_client.ainfer(captions)
    if dense_vecs is None:
        raise RuntimeError("Failed to embed event captions")

    for node, vec in zip(event_nodes, dense_vecs):
        node.embedding = vec

    return event_nodes


def build_event_entity_links(event_nodes: list[EventNode]) -> list[EventEntityEdge]:
    edges: list[EventEntityEdge] = []
    for ev in event_nodes:
        for global_id in ev.entities_global:
            edges.append(EventEntityEdge(
                video_id=ev.video_id,
                from_key=f"events/{ev.key}",
                to_key=f"entities/{global_id}",
            ))
    return edges


def build_event_sim_matrix(event_nodes: list[EventNode]) -> np.ndarray:
    vecs = np.array([e.embedding for e in event_nodes], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vecs /= norms
    return cosine_similarity(vecs)


async def llm_confirm_link(
    event_a: EventNode,
    event_b: EventNode,
    structured_llm,
    semaphore: asyncio.Semaphore,
    cost_tracker: CostTracker,
) -> EventLinkOutput | None:
    prompt = f"""You are a video understanding expert reviewing two segments from the same video.
Decide if they should be linked in a knowledge graph, and if so, how.

Event A [t={event_a.start_time}-{event_a.end_time}]:
{event_a.caption}

Event B [t={event_b.start_time}-{event_b.end_time}]:
{event_b.caption}

Shared entities: {list(set(event_a.entities_global) & set(event_b.entities_global))}

Link types:
- SEMANTICALLY_SIMILAR : same topic / theme
- SHARES_CONTEXT       : same physical entities involved
- CAUSAL               : A causes or directly leads to B
- UNRELATED            : no meaningful connection

Return JSON with: should_link (bool), link_type (str), reason (str).
"""
    msg = ChatMessage(blocks=[TextBlock(text=prompt)])
    async with semaphore:
        try:
            response = await structured_llm.achat([msg])

            prompt_tokens = response.additional_kwargs.get('prompt_tokens', 0) or 0
            completion_tokens = response.additional_kwargs.get('completion_tokens', 0) or 0
            cost = response.additional_kwargs.get('cost', 0.0) or 0.0
            cost_tracker.add_usage(prompt_tokens, completion_tokens, cost)

            return cast(EventLinkOutput, response.raw)
        except Exception as ex:
            print(f"  [WARN] LLM link error ({event_a.key} ↔ {event_b.key}): {ex}")
            return None


async def build_event_edges(
    event_nodes: list[EventNode],
    event_sim_matrix: np.ndarray,
    *,
    semantic_threshold: float,
    llm_confirm_threshold: float,
    jaccard_threshold: float,
    llm_client,
    max_concurrent_llm: int,
    cost_tracker: CostTracker,
) -> list[EventEdge]:
    N = len(event_nodes)
    edges: list[EventEdge] = []
    linked_pairs: set[tuple[str, str]] = set()

    for i in range(N - 1):
        a, b = event_nodes[i], event_nodes[i + 1]
        edges.append(EventEdge(
            video_id=a.video_id,
            from_key=f"events/{a.key}",
            to_key=f"events/{b.key}",
            edge_type="NEXT_EVENT",
            temporal_gap_s=_time_gap(a.end_sec, b.start_sec),
            shared_entities=len(set(a.entities_global) & set(b.entities_global)),
        ))
        linked_pairs.add((a.key, b.key))

    print(f"  Pass 1 NEXT_EVENT edges             : {len(edges)}")

    llm_candidates: list[tuple[int, int, float]] = []
    for i in range(N):
        for j in range(i + 1, N):
            sim = float(event_sim_matrix[i, j])
            pair = (event_nodes[i].key, event_nodes[j].key)
            if pair in linked_pairs:
                continue
            if sim >= semantic_threshold:
                edges.append(EventEdge(
                    video_id=event_nodes[i].video_id,
                    from_key=f"events/{pair[0]}",
                    to_key=f"events/{pair[1]}",
                    edge_type="SEMANTICALLY_SIMILAR",
                    temporal_gap_s=_time_gap(event_nodes[i].end_sec, event_nodes[j].start_sec),
                    similarity=round(sim, 4),
                    shared_entities=len(
                        set(event_nodes[i].entities_global) & set(event_nodes[j].entities_global)
                    ),
                ))
                linked_pairs.add(pair)
            elif sim >= llm_confirm_threshold:
                llm_candidates.append((i, j, sim))

    sem_sim_count = sum(1 for e in edges if e.edge_type == "SEMANTICALLY_SIMILAR")
    print(f"  Pass 2 SEMANTICALLY_SIMILAR edges   : {sem_sim_count}")
    print(f"         LLM borderline candidates     : {len(llm_candidates)}")

    shares_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            pair = (event_nodes[i].key, event_nodes[j].key)
            if pair in linked_pairs:
                continue
            set_a = set(event_nodes[i].entities_global)
            set_b = set(event_nodes[j].entities_global)
            union = set_a | set_b
            if not union:
                continue
            jaccard = len(set_a & set_b) / len(union)
            if jaccard >= jaccard_threshold:
                edges.append(EventEdge(
                    video_id=event_nodes[i].video_id,
                    from_key=f"events/{pair[0]}",
                    to_key=f"events/{pair[1]}",
                    edge_type="SHARES_CONTEXT",
                    temporal_gap_s=_time_gap(event_nodes[i].end_sec, event_nodes[j].start_sec),
                    similarity=round(float(event_sim_matrix[i, j]), 4),
                    shared_entities=len(set_a & set_b),
                    jaccard=round(jaccard, 4),
                ))
                linked_pairs.add(pair)
                shares_count += 1

    print(f"  Pass 3 SHARES_CONTEXT edges         : {shares_count}")

    if llm_candidates:
        structured_link_llm = llm_client.as_structured_llm(EventLinkOutput)
        semaphore = asyncio.Semaphore(max_concurrent_llm)

        filtered_candidates = [
            (i, j, sim)
            for i, j, sim in llm_candidates
            if (event_nodes[i].key, event_nodes[j].key) not in linked_pairs
        ]

        async def _confirm(i: int, j: int, sim: float) -> EventEdge | None:
            result = await llm_confirm_link(event_nodes[i], event_nodes[j], structured_link_llm, semaphore, cost_tracker)
            if result and result.should_link and result.link_type != "UNRELATED":
                pair = (event_nodes[i].key, event_nodes[j].key)
                return EventEdge(
                    video_id=event_nodes[i].video_id,
                    from_key=f"events/{pair[0]}",
                    to_key=f"events/{pair[1]}",
                    edge_type=result.link_type,
                    temporal_gap_s=_time_gap(event_nodes[i].end_sec, event_nodes[j].start_sec),
                    similarity=round(sim, 4),
                    shared_entities=len(
                        set(event_nodes[i].entities_global) & set(event_nodes[j].entities_global)
                    ),
                    llm_reason=result.reason,
                    llm_confirmed=True,
                )
            return None

        tasks = [_confirm(i, j, sim) for i, j, sim in filtered_candidates]
        llm_edges: list[EventEdge] = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="  LLM event linking"):
            result = await coro
            if result:
                linked_pairs.add((
                    result.from_key.split("/")[1],
                    result.to_key.split("/")[1],
                ))
                llm_edges.append(result)

        edges.extend(llm_edges)
        print(f"  Pass 4 LLM-confirmed edges          : {len(llm_edges)}")
    else:
        print("  Pass 4 LLM-confirmed edges          : 0 (no candidates)")

    print(f"  Total event edges                   : {len(edges)}")
    return edges



async def build_micro_event_nodes(
    event_nodes: list[EventNode],
    dense_client: MMBertClient,
) -> list[MicroEventNode]:
    """Promote each bare micro-event string into a proper MicroEventNode."""
    nodes: list[MicroEventNode] = []

    for ev in event_nodes:
        for m_idx, text in enumerate(ev.micro_events):
            nodes.append(MicroEventNode(
                video_id=ev.video_id,
                key=f"micro_{ev.segment_index:04d}_{m_idx:02d}",
                parent_event_key=f"events/{ev.key}",
                segment_index=ev.segment_index,
                micro_index=m_idx,
                start_time=ev.start_time,
                related_caption_context=ev.caption,
                end_time=ev.end_time,
                text=text,
                start_secs=ev.start_sec,
                end_secs=ev.end_sec,
                entities_global=ev.entities_global,
                embedding=None,
            ))

    if not nodes:
        print("  No micro-events found — skipping embedding.")
        return nodes

    print(f"  Embedding {len(nodes)} micro-event texts...")
    texts = [n.text for n in nodes]
    dense_vecs = await dense_client.ainfer(texts)
    if dense_vecs is None:
        raise RuntimeError("Failed to embed micro-event texts")
    for node, vec in zip(nodes, dense_vecs):
        node.embedding = vec

    return nodes


def _build_micro_sim_matrix(micro_nodes: list[MicroEventNode]) -> np.ndarray:
    vecs = np.array([n.embedding for n in micro_nodes], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vecs /= norms
    return cosine_similarity(vecs)


async def _llm_confirm_micro_link(
    micro_a: MicroEventNode,
    micro_b: MicroEventNode,
    structured_llm,
    semaphore: asyncio.Semaphore,
    cost_tracker: CostTracker,
) -> MicroLinkOutput | None:
    shared = list(set(micro_a.entities_global) & set(micro_b.entities_global))
    prompt = f"""You are a video understanding expert.
Decide if two atomic micro-events from the same video should be linked in a knowledge graph.

Micro-event A [segment {micro_a.segment_index}, t={micro_a.start_time}-{micro_a.end_time}]:
{micro_a.text}

Micro-event B [segment {micro_b.segment_index}, t={micro_b.start_time}-{micro_b.end_time}]:
{micro_b.text}

Shared entities: {shared}

Link types:
- SEMANTICALLY_SIMILAR_MICRO : same action / topic at the micro level
- SHARES_CONTEXT_MICRO       : same physical entities / props are involved
- CAUSAL_MICRO               : micro-event A directly causes micro-event B
- UNRELATED                  : no meaningful connection

Return JSON with: should_link (bool), link_type (str), reason (str).
"""
    msg = ChatMessage(blocks=[TextBlock(text=prompt)])
    async with semaphore:
        try:
            response = await structured_llm.achat([msg])

            # Extract usage from raw_response (the original ChatResponse)
            if hasattr(response, 'raw_response') and response.raw_response is not None:
                raw_resp = response.raw_response
                if hasattr(raw_resp, 'raw') and raw_resp.raw and hasattr(raw_resp.raw, 'usage'):
                    usage = raw_resp.raw.usage
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
                    completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
                    cost = getattr(usage, 'cost', 0.0) or 0.0
                    cost_tracker.add_usage(prompt_tokens, completion_tokens, cost)
                elif hasattr(raw_resp, 'additional_kwargs'):
                    # Fallback to additional_kwargs if raw.usage not available
                    prompt_tokens = raw_resp.additional_kwargs.get('prompt_tokens', 0) or 0
                    completion_tokens = raw_resp.additional_kwargs.get('completion_tokens', 0) or 0
                    cost = raw_resp.additional_kwargs.get('cost', 0.0) or 0.0
                    cost_tracker.add_usage(prompt_tokens, completion_tokens, cost)

            return cast(MicroLinkOutput, response.raw)
        except Exception as ex:
            print(f"  [WARN] Micro LLM link error ({micro_a.key} ↔ {micro_b.key}): {ex}")
            return None


async def build_micro_event_edges(
    micro_nodes: list[MicroEventNode],
    micro_sim_matrix: np.ndarray,
    *,
    window_size: int,
    semantic_threshold: float,
    llm_confirm_threshold: float,
    jaccard_threshold: float,
    llm_client,
    max_concurrent_llm: int,
    cost_tracker: CostTracker,
) -> list[MicroEventEdge]:
    if not micro_nodes:
        return []

    by_seg: dict[int, list[tuple[int, MicroEventNode]]] = defaultdict(list)
    node_to_flat_idx: dict[str, int] = {}
    for flat_idx, mn in enumerate(micro_nodes):
        by_seg[mn.segment_index].append((flat_idx, mn))
        node_to_flat_idx[mn.key] = flat_idx

    seg_indices = sorted(by_seg.keys())
    edges: list[MicroEventEdge] = []
    linked_pairs: set[tuple[str, str]] = set()

    def _cross_pairs(seg_i: int, seg_j: int):
        for fi, ma in by_seg[seg_i]:
            for fj, mb in by_seg[seg_j]:
                pair = (ma.key, mb.key)
                if pair not in linked_pairs:
                    yield fi, fj, ma, mb, pair

    llm_candidates: list[tuple[int, int, float]] = []
    sem_sim_count = 0

    for s_pos, seg_i in enumerate(seg_indices):
        for seg_j in seg_indices[s_pos + 1: s_pos + 1 + window_size]:
            for fi, fj, ma, mb, pair in _cross_pairs(seg_i, seg_j):
                sim = float(micro_sim_matrix[fi, fj])
                if sim >= semantic_threshold:
                    edges.append(MicroEventEdge(
                        video_id=ma.video_id,
                        from_key=f"micro_events/{ma.key}",
                        to_key=f"micro_events/{mb.key}",
                        edge_type="SEMANTICALLY_SIMILAR_MICRO",
                        temporal_gap_s=_time_gap(ma.end_secs, mb.start_secs),
                        similarity=round(sim, 4),
                        shared_entities=len(set(ma.entities_global) & set(mb.entities_global)),
                    ))
                    linked_pairs.add(pair)
                    sem_sim_count += 1
                elif sim >= llm_confirm_threshold:
                    llm_candidates.append((fi, fj, sim))

    print(f"  Micro Pass 2 SEMANTICALLY_SIMILAR_MICRO: {sem_sim_count}")
    print(f"         LLM borderline candidates        : {len(llm_candidates)}")

    # Pass 3: SHARES_CONTEXT_MICRO (Jaccard)
    shares_count = 0
    for s_pos, seg_i in enumerate(seg_indices):
        for seg_j in seg_indices[s_pos + 1: s_pos + 1 + window_size]:
            for _, _, ma, mb, pair in _cross_pairs(seg_i, seg_j):
                if pair in linked_pairs:
                    continue
                set_a = set(ma.entities_global)
                set_b = set(mb.entities_global)
                union = set_a | set_b
                if not union:
                    continue
                jaccard = len(set_a & set_b) / len(union)
                if jaccard >= jaccard_threshold:
                    fi = node_to_flat_idx[ma.key]
                    fj = node_to_flat_idx[mb.key]
                    edges.append(MicroEventEdge(
                        video_id=ma.video_id,
                        from_key=f"micro_events/{ma.key}",
                        to_key=f"micro_events/{mb.key}",
                        edge_type="SHARES_CONTEXT_MICRO",
                        temporal_gap_s=_time_gap(ma.end_secs, mb.start_secs),
                        similarity=round(float(micro_sim_matrix[fi, fj]), 4),
                        shared_entities=len(set_a & set_b),
                        jaccard=round(jaccard, 4),
                    ))
                    linked_pairs.add(pair)
                    shares_count += 1

    print(f"  Micro Pass 3 SHARES_CONTEXT_MICRO      : {shares_count}")

    # Pass 4: LLM-confirmed micro links
    if llm_candidates:
        structured_micro_llm = llm_client.as_structured_llm(MicroLinkOutput)
        semaphore = asyncio.Semaphore(max_concurrent_llm)

        filtered = [
            (fi, fj, sim)
            for fi, fj, sim in llm_candidates
            if (micro_nodes[fi].key, micro_nodes[fj].key) not in linked_pairs
        ]

        async def _confirm_micro(fi: int, fj: int, sim: float) -> MicroEventEdge | None:
            ma, mb = micro_nodes[fi], micro_nodes[fj]
            result = await _llm_confirm_micro_link(ma, mb, structured_micro_llm, semaphore, cost_tracker)
            if result and result.should_link and result.link_type != "UNRELATED":
                return MicroEventEdge(
                    video_id=ma.video_id,
                    from_key=f"micro_events/{ma.key}",
                    to_key=f"micro_events/{mb.key}",
                    edge_type=result.link_type,
                    temporal_gap_s=_time_gap(ma.end_secs, mb.start_secs),
                    similarity=round(sim, 4),
                    shared_entities=len(set(ma.entities_global) & set(mb.entities_global)),
                    llm_reason=result.reason,
                    llm_confirmed=True,
                )
            return None

        tasks = [_confirm_micro(fi, fj, sim) for fi, fj, sim in filtered]
        llm_micro_edges: list[MicroEventEdge] = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="  LLM micro linking"):
            result = await coro
            if result:
                linked_pairs.add((
                    result.from_key.split("/")[1],
                    result.to_key.split("/")[1],
                ))
                llm_micro_edges.append(result)

        edges.extend(llm_micro_edges)
        print(f"  Micro Pass 4 LLM-confirmed            : {len(llm_micro_edges)}")
    else:
        print("  Micro Pass 4 LLM-confirmed            : 0 (no candidates)")

    print(f"  Total micro-event edges               : {len(edges)}")
    return edges


async def run_event_linking(
    resolved_kg: ResolvedKG,
    dense_client: MMBertClient,
    llm_client,
    semantic_threshold: float = 0.80,
    llm_confirm_threshold: float = 0.60,
    jaccard_threshold: float = 0.30,
    micro_window_size: int = 2,
    micro_semantic_threshold: float = 0.85,
    micro_llm_confirm_threshold: float = 0.65,
    micro_jaccard_threshold: float = 0.40,
    max_concurrent_llm: int = 5,
    cost_tracker: CostTracker | None = None,
) -> EnhancedKG:
    """Run the full event linking pipeline."""
    if cost_tracker is None:
        cost_tracker = CostTracker()

    print(f"\n[Stage 3] Event & Micro-Event Linking")
    print(f"  segments: {len(resolved_kg.segments)} | entities: {len(resolved_kg.entities)} | relationships: {len(resolved_kg.relationships)}")

    # Build & embed event nodes
    print(f"  Building & embedding event documents...")
    event_nodes = await build_event_docs(resolved_kg, dense_client)
    print(f"  {len(event_nodes)} event documents created.")

    # Event-entity links
    event_entity_links = build_event_entity_links(event_nodes)
    print(f"  {len(event_entity_links)} event-entity links created.")

    # Event-level edges (4 passes)
    print(f"  Building event sequence edges (4 passes)...")
    event_sim_matrix = build_event_sim_matrix(event_nodes)
    event_edges = await build_event_edges(
        event_nodes,
        event_sim_matrix,
        semantic_threshold=semantic_threshold,
        llm_confirm_threshold=llm_confirm_threshold,
        jaccard_threshold=jaccard_threshold,
        llm_client=llm_client,
        max_concurrent_llm=max_concurrent_llm,
        cost_tracker=cost_tracker,
    )

    # Build & embed micro-event nodes
    print(f"  Building & embedding micro-event nodes...")
    micro_nodes = await build_micro_event_nodes(event_nodes, dense_client)
    print(f"  {len(micro_nodes)} micro-event nodes created.")

    # Micro-event edges (sliding window, 4 passes)
    print(f"  Building micro-event edges (window={micro_window_size}, 4 passes)...")
    micro_event_edges: list[MicroEventEdge] = []
    if micro_nodes:
        micro_sim_matrix = _build_micro_sim_matrix(micro_nodes)
        micro_event_edges = await build_micro_event_edges(
            micro_nodes,
            micro_sim_matrix,
            window_size=micro_window_size,
            semantic_threshold=micro_semantic_threshold,
            llm_confirm_threshold=micro_llm_confirm_threshold,
            jaccard_threshold=micro_jaccard_threshold,
            llm_client=llm_client,
            max_concurrent_llm=max_concurrent_llm,
            cost_tracker=cost_tracker,
        )
    else:
        print("  No micro-event nodes — skipping edge building.")

    return EnhancedKG(
        video_id=resolved_kg.video_id,
        entities=resolved_kg.entities,
        relationships=resolved_kg.relationships,
        segments=resolved_kg.segments,
        events=event_nodes,
        event_entity_links=event_entity_links,
        event_edges=event_edges,
        micro_event_nodes=micro_nodes,
        micro_event_edges=micro_event_edges,
    )