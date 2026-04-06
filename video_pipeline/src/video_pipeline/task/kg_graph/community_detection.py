import asyncio
from typing import Any

import igraph as ig
import leidenalg
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage

from video_pipeline.core.client.inference import MMBertClient

from .models import (
    EnhancedKG,
    CommunityDoc,
    MembershipEdge,
    EventCommunityEdge,
    GraphStats,
    CommunitiesOutput,
    CostTracker,
)


class CommunitySummary(BaseModel):
    title: str = Field(..., description="Short 4-6 word title for this community.")
    summary: str = Field(..., description="2-3 sentence description of what this entity cluster represents.")


def build_igraph(enhanced_kg: EnhancedKG) -> tuple[ig.Graph, list[str], dict[str, int]]:
    node_list = [e.global_entity_id for e in enhanced_kg.entities]
    node_idx = {k: i for i, k in enumerate(node_list)}

    ig_edges: list[tuple[int, int]] = []
    ig_weights: list[float] = []

    for rel in enhanced_kg.relationships:
        from_key = rel.subject_global
        to_key = rel.object_global
        weight = float(rel.weight)

        if from_key in node_idx and to_key in node_idx:
            ig_edges.append((node_idx[from_key], node_idx[to_key]))
            ig_weights.append(weight)

    G = ig.Graph(n=len(node_list), edges=ig_edges, directed=False)
    G.es["weight"] = ig_weights
    G.vs["name"] = node_list
    G = G.simplify(combine_edges={"weight": "sum"})

    print(f"  Graph: {G.vcount()} nodes, {G.ecount()} edges")
    return G, node_list, node_idx


def run_leiden(
    G: ig.Graph,
    node_list: list[str],
    n_iterations: int = 10,
    seed: int = 42,
) -> tuple[list[dict], Any]:
    partition = leidenalg.find_partition(
        G,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        n_iterations=n_iterations,
        seed=seed,
    )
    modularity = partition.modularity
    print(f"  Detected {len(partition)} communities  (modularity={modularity:.4f})")

    community_data: list[dict] = []
    for i, community in enumerate(partition):
        member_keys = [node_list[v] for v in community]
        community_data.append({"comm_idx": i, "member_keys": member_keys})

    return community_data, modularity


async def summarize_community(
    community_id: int,
    member_entities: list[Any],
    structured_llm,
    semaphore: asyncio.Semaphore,
    cost_tracker: CostTracker,
) -> dict:
    entity_lines = [
        f"- {e.entity_name} ({e.entity_type}): {e.desc[:120]}"
        for e in member_entities
    ]
    prompt = f"""You are analysing a Knowledge Graph extracted from a video.
The following entities form a tightly connected cluster (Community {community_id}).
Summarise what this community represents in the video.

Entities:
{chr(10).join(entity_lines)}

Output a JSON object with exactly this structure (no trailing commas):
{{
  "title": "Short 4-6 word title",
  "summary": "2-3 sentence description of what this entity cluster represents."
}}

Fields:
- title: a concise 4-6 word title for this community
- summary: a 2-3 sentence description of what this entity cluster represents
"""
    msg = HumanMessage(content=prompt)
    async with semaphore:
        try:
            result, usage = await structured_llm([msg])

            prompt_tokens = usage.get('prompt_tokens', 0) or 0
            completion_tokens = usage.get('completion_tokens', 0) or 0
            cost = usage.get('cost', 0.0) or 0.0
            cost_tracker.add_usage(prompt_tokens, completion_tokens, cost)

            return {"title": result.title, "summary": result.summary}
        except Exception as ex:
            print(f"  [WARN] Community {community_id} summary error: {ex}")
            return {"title": f"Community {community_id}", "summary": ""}


async def run_llm_summaries(
    community_data: list[dict],
    enhanced_kg: EnhancedKG,
    llm_client,
    max_concurrent_llm: int,
    cost_tracker: CostTracker,
) -> list[dict]:
    global_entity_map = {e.global_entity_id: e for e in enhanced_kg.entities}

    structured_llm = llm_client.as_structured_llm(CommunitySummary)
    semaphore = asyncio.Semaphore(max_concurrent_llm)

    tasks = [
        summarize_community(
            c["comm_idx"],
            [global_entity_map[k] for k in c["member_keys"] if k in global_entity_map],
            structured_llm,
            semaphore,
            cost_tracker,
        )
        for c in community_data
    ]
    print(f"  Community summaries: {len(tasks)}")
    return list(await asyncio.gather(*tasks))


async def assemble_output(
    video_id: str,
    community_data: list[dict],
    summaries: list[dict],
    dense_client: MMBertClient,
    graph_stats: GraphStats,
) -> CommunitiesOutput:
    summary_texts = [s["title"] + ". " + s["summary"] for s in summaries]
    print(f"  Embedding {len(summary_texts)} community summaries...")
    comm_embeddings = await dense_client.ainfer(summary_texts)
    if comm_embeddings is None:
        raise RuntimeError("Failed to embed community summaries")

    communities: list[CommunityDoc] = []
    membership_edges: list[MembershipEdge] = []

    for comm_info, comm_sum, comm_emb in zip(community_data, summaries, comm_embeddings):
        comm_idx = comm_info["comm_idx"]
        comm_key = f"comm_{comm_idx:04d}"

        communities.append(CommunityDoc(
            video_id=video_id,
            comm_key=comm_key,
            comm_idx=comm_idx,
            title=comm_sum["title"],
            summary=comm_sum["summary"],
            size=len(comm_info["member_keys"]),
            member_keys=comm_info["member_keys"],
            event_keys=[],
            embedding=comm_emb,
        ))

        for member_key in comm_info["member_keys"]:
            membership_edges.append(MembershipEdge(
                video_id=video_id,
                from_key=f"entities/{member_key}",
                to_key=f"communities/{comm_key}",
            ))

    return CommunitiesOutput(
        video_id=video_id,
        communities=communities,
        membership_edges=membership_edges,
        event_community_edges=[],
        graph_stats=graph_stats,
    )


def assign_events_to_communities(
    output: CommunitiesOutput,
    enhanced_kg: EnhancedKG,
) -> CommunitiesOutput:
    """Assign each event to the community whose member entities it overlaps most."""
    comm_entity_sets: dict[str, set[str]] = {
        c.comm_key: set(c.member_keys) for c in output.communities
    }

    events = enhanced_kg.events
    if not events:
        print("  No events found — skipping event→community assignment.")
        return output

    event_assignment: dict[str, str] = {}
    event_shared: dict[str, int] = {}
    event_method: dict[str, str] = {}

    zero_overlap_events: list[Any] = []

    for ev in events:
        ev_entities = set(ev.entities_global)
        if not ev_entities:
            zero_overlap_events.append(ev)
            continue

        votes = {
            comm_key: len(ev_entities & members)
            for comm_key, members in comm_entity_sets.items()
        }
        best_comm = max(votes, key=lambda k: votes[k])
        best_score = votes[best_comm]

        if best_score == 0:
            zero_overlap_events.append(ev)
        else:
            event_assignment[ev.key] = best_comm
            event_shared[ev.key] = best_score
            event_method[ev.key] = "majority_vote"

    print(
        f"  Majority-vote assignments : {len(event_assignment)}  "
        f"| zero-overlap fallbacks needed: {len(zero_overlap_events)}"
    )

    sorted_events = sorted(events, key=lambda e: e.start_sec)
    sorted_keys = [e.key for e in sorted_events]
    sorted_secs = [e.start_sec for e in sorted_events]

    for ev in zero_overlap_events:
        best_dist = float("inf")
        best_fallback: str | None = None

        for other_key, other_sec in zip(sorted_keys, sorted_secs):
            if other_key not in event_assignment:
                continue
            dist = abs(other_sec - ev.start_sec)
            if dist < best_dist:
                best_dist = dist
                best_fallback = other_key

        if best_fallback is not None:
            event_assignment[ev.key] = event_assignment[best_fallback]
            event_shared[ev.key] = 0
            event_method[ev.key] = "temporal_fallback"
        else:
            fallback_key = output.communities[0].comm_key if output.communities else "comm_0000"
            event_assignment[ev.key] = fallback_key
            event_shared[ev.key] = 0
            event_method[ev.key] = "default_fallback"

    comm_key_to_idx = {c.comm_key: i for i, c in enumerate(output.communities)}
    event_community_edges: list[EventCommunityEdge] = []

    for ev_key, comm_key in event_assignment.items():
        event_community_edges.append(EventCommunityEdge(
            video_id=output.video_id,
            from_key=f"events/{ev_key}",
            to_key=f"communities/{comm_key}",
            shared_entities=event_shared.get(ev_key, 0),
            assignment=event_method.get(ev_key, "unknown"),
        ))
        idx = comm_key_to_idx.get(comm_key)
        if idx is not None:
            output.communities[idx].event_keys.append(ev_key)

    output.event_community_edges = event_community_edges

    fallback_count = sum(1 for m in event_method.values() if "fallback" in m)
    print(f"  Total event→community edges : {len(event_community_edges)}")
    print(f"  Temporal / default fallbacks: {fallback_count}")

    return output


async def run_community_detection(
    enhanced_kg: EnhancedKG,
    dense_client: MMBertClient,
    llm_client,
    n_iterations: int = 10,
    seed: int = 42,
    max_concurrent_llm: int = 5,
    cost_tracker: CostTracker | None = None,
) -> CommunitiesOutput:
    """Run the full community detection pipeline."""
    if cost_tracker is None:
        cost_tracker = CostTracker()

    print(f"\n[Stage 4] Community Detection")
    print(f"  entities={len(enhanced_kg.entities)}  relationships={len(enhanced_kg.relationships)}  events={len(enhanced_kg.events)}")

    print(f"  Building igraph...")
    G, node_list, _ = build_igraph(enhanced_kg)

    print(f"  Running Leiden community detection...")
    community_data, modularity = run_leiden(G, node_list, n_iterations=n_iterations, seed=seed)

    graph_stats = GraphStats(
        n_nodes=G.vcount(),
        n_edges=G.ecount(),
        n_communities=len(community_data),
        modularity=round(modularity, 6),
    )

    print(f"  Generating LLM community summaries...")
    summaries = await run_llm_summaries(community_data, enhanced_kg, llm_client, max_concurrent_llm, cost_tracker)
    output = await assemble_output(enhanced_kg.video_id, community_data, summaries, dense_client, graph_stats)
    print(f"  Assigning events to communities (majority vote)...")
    output = assign_events_to_communities(output, enhanced_kg)
    print(f"  communities: {len(output.communities)}")
    print(f"  membership edges: {len(output.membership_edges)}")
    print(f"  event-community edges: {len(output.event_community_edges)}")

    return output