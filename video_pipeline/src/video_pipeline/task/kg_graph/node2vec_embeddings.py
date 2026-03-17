"""Node2Vec Embeddings - Stage 5.

Train Node2Vec structural embeddings on three graph variants.
"""

from __future__ import annotations

import warnings

import networkx as nx

warnings.filterwarnings("ignore")

from .models import (
    EnhancedKG,
    CommunitiesOutput,
    Node2VecMeta,
    NodeEmbedding,
    Node2VecOutput,
)
from .node2vec import Node2Vec

def build_entity_only_graph(enhanced_kg: EnhancedKG) -> nx.Graph:
    """Graph A — entity nodes + entity↔entity relationship edges."""
    G = nx.Graph()

    for entity in enhanced_kg.entities:
        G.add_node(entity.global_entity_id, node_type="entity", label=entity.entity_name)

    for rel in enhanced_kg.relationships:
        subj = rel.subject_global
        obj = rel.object_global
        w = float(rel.weight)
        if subj and obj and subj != obj and G.has_node(subj) and G.has_node(obj):
            if G.has_edge(subj, obj):
                G[subj][obj]["weight"] += w
            else:
                G.add_edge(subj, obj, weight=w)

    print(f"  [A] entity_only          : {G.number_of_nodes():>5} nodes, {G.number_of_edges():>6} edges")
    return G


def build_entity_micro_event_graph(enhanced_kg: EnhancedKG) -> nx.Graph:
    """Graph B — entity + micro-event nodes."""
    G = nx.Graph()

    for entity in enhanced_kg.entities:
        G.add_node(entity.global_entity_id, node_type="entity", label=entity.entity_name)

    for mn in enhanced_kg.micro_event_nodes:
        G.add_node(mn.key, node_type="micro_event", label=mn.text[:60])

    for rel in enhanced_kg.relationships:
        subj = rel.subject_global
        obj = rel.object_global
        w = float(rel.weight)
        if subj and obj and subj != obj and G.has_node(subj) and G.has_node(obj):
            if G.has_edge(subj, obj):
                G[subj][obj]["weight"] += w
            else:
                G.add_edge(subj, obj, weight=w)

    for mn in enhanced_kg.micro_event_nodes:
        for gid in mn.entities_global:
            if G.has_node(gid) and G.has_node(mn.key):
                G.add_edge(gid, mn.key, weight=1.0)

    MICRO_WEIGHTS = {
        "NEXT_MICRO_EVENT": 2.0,
        "SEMANTICALLY_SIMILAR_MICRO": 1.5,
        "SHARES_CONTEXT_MICRO": 1.2,
        "CAUSAL_MICRO": 1.8,
    }
    for edge in enhanced_kg.micro_event_edges:
        from_key = edge.from_key.split("/")[-1]
        to_key = edge.to_key.split("/")[-1]
        base_w = MICRO_WEIGHTS.get(edge.edge_type, 1.0)
        w = (base_w * edge.similarity) if edge.similarity is not None else base_w
        if G.has_node(from_key) and G.has_node(to_key):
            if G.has_edge(from_key, to_key):
                G[from_key][to_key]["weight"] = max(G[from_key][to_key]["weight"], w)
            else:
                G.add_edge(from_key, to_key, weight=w)

    big_event_keys = {ev.key for ev in enhanced_kg.events}
    for mn in enhanced_kg.micro_event_nodes:
        parent = mn.parent_event_key.split("/")[-1]
        if parent in big_event_keys:
            if not G.has_node(parent):
                G.add_node(parent, node_type="event", label=parent)
            G.add_edge(mn.key, parent, weight=1.5)

    print(f"  [B] entity_micro_event   : {G.number_of_nodes():>5} nodes, {G.number_of_edges():>6} edges")
    return G


def build_full_heterogeneous_graph(
    enhanced_kg: EnhancedKG,
    communities: CommunitiesOutput,
) -> nx.Graph:
    G = build_entity_micro_event_graph(enhanced_kg)

    for event in enhanced_kg.events:
        if not G.has_node(event.key):
            G.add_node(event.key, node_type="event", label=event.caption[:60])

    for link in enhanced_kg.event_entity_links:
        ev_key = link.from_key.split("/")[-1]
        entity_key = link.to_key.split("/")[-1]
        if G.has_node(ev_key) and G.has_node(entity_key):
            if not G.has_edge(ev_key, entity_key):
                G.add_edge(ev_key, entity_key, weight=1.0)

    BIG_EVENT_WEIGHTS = {
        "NEXT_EVENT": 2.0,
        "SEMANTICALLY_SIMILAR": 1.5,
        "SHARES_CONTEXT": 1.2,
        "CAUSAL": 1.8,
    }
    for edge in enhanced_kg.event_edges:
        from_key = edge.from_key.split("/")[-1]
        to_key = edge.to_key.split("/")[-1]
        base_w = BIG_EVENT_WEIGHTS.get(edge.edge_type, 1.0)
        w = (base_w * edge.similarity) if edge.similarity is not None else base_w
        if G.has_node(from_key) and G.has_node(to_key):
            if G.has_edge(from_key, to_key):
                G[from_key][to_key]["weight"] = max(G[from_key][to_key]["weight"], w)
            else:
                G.add_edge(from_key, to_key, weight=w)

    for comm in communities.communities:
        G.add_node(comm.comm_key, node_type="community", label=comm.title)

    for edge in communities.membership_edges:
        entity_key = edge.from_key.split("/")[-1]
        comm_key = edge.to_key.split("/")[-1]
        if G.has_node(entity_key) and G.has_node(comm_key):
            G.add_edge(entity_key, comm_key, weight=1.0)

    for edge in communities.event_community_edges:
        ev_key = edge.from_key.split("/")[-1]
        comm_key = edge.to_key.split("/")[-1]
        w = max(1.0, float(edge.shared_entities))
        if G.has_node(ev_key) and G.has_node(comm_key):
            if G.has_edge(ev_key, comm_key):
                G[ev_key][comm_key]["weight"] = max(G[ev_key][comm_key]["weight"], w)
            else:
                G.add_edge(ev_key, comm_key, weight=w)

    print(f"  [C] full_heterogeneous   : {G.number_of_nodes():>5} nodes, {G.number_of_edges():>6} edges")
    return G


def relabel_graph_to_integers(G: nx.Graph) -> tuple[nx.Graph, dict[int, str]]:
    """Relabel graph nodes to consecutive integers for Node2Vec.

    Returns:
        relabeled_graph: Graph with integer node labels (0 to n-1)
        id_map: Mapping from integer index to original node ID
    """
    nodes = list(G.nodes())
    id_map = {i: node for i, node in enumerate(nodes)}
    reverse_map = {node: i for i, node in enumerate(nodes)}

    H = nx.Graph()
    for i, node in enumerate(nodes):
        H.add_node(i, **G.nodes[node])

    for u, v, data in G.edges(data=True):
        H.add_edge(reverse_map[u], reverse_map[v], **data)

    return H, id_map


def train_node2vec(
    G: nx.Graph,
    graph_name: str,
    *,
    dim: int,
    walk_length: int,
    num_walks: int,
    p: float,
    q: float,
    window: int,
    workers: int,
    seed: int,
) -> dict[str, list[float]]:
    """Train Node2Vec on a graph and return embeddings keyed by original node IDs."""
    print(f"    Training Node2Vec ({graph_name})...")

    if G.number_of_nodes() == 0:
        return {}

    G_int, id_map = relabel_graph_to_integers(G)

    model = Node2Vec(
        walk_number=num_walks,
        walk_length=walk_length,
        p=p,
        q=q,
        dimensions=dim,
        workers=workers,
        window_size=window,
        seed=seed,
    )

    model.fit(G_int)
    embeddings = model.get_embedding()

    output = {}
    for i in range(len(id_map)):
        original_id = id_map[i]
        output[str(original_id)] = embeddings[i].tolist()

    print(f"      trained on {G.number_of_nodes()} nodes, embedding dim={dim}")
    return output


def assemble_output(
    enhanced_kg: EnhancedKG,
    communities: CommunitiesOutput,
    emb_A: dict[str, list[float]],
    emb_B: dict[str, list[float]],
    emb_C: dict[str, list[float]],
    dim: int,
    walk_length: int,
    num_walks: int,
    p: float,
    q: float,
    window: int,
) -> Node2VecOutput:
    nodes: dict[str, NodeEmbedding] = {}

    for entity in enhanced_kg.entities:
        key = entity.global_entity_id
        nodes[key] = NodeEmbedding(
            video_id=enhanced_kg.video_id,
            node_type="entity",
            label=entity.entity_name,
            entity_only_embedding=emb_A.get(key),
            entity_event_embedding=emb_B.get(key),
            full_heterogeneous_embedding=emb_C.get(key),
        )

    for mn in enhanced_kg.micro_event_nodes:
        key = mn.key
        nodes[key] = NodeEmbedding(
            video_id=enhanced_kg.video_id,
            node_type="micro_event",
            label=mn.text[:80],
            entity_only_embedding=None,
            entity_event_embedding=emb_B.get(key),
            full_heterogeneous_embedding=emb_C.get(key),
        )

    for event in enhanced_kg.events:
        key = event.key
        nodes[key] = NodeEmbedding(
            video_id=enhanced_kg.video_id,
            node_type="event",
            label=event.caption[:80],
            entity_only_embedding=None,
            entity_event_embedding=emb_B.get(key),
            full_heterogeneous_embedding=emb_C.get(key),
        )

    for comm in communities.communities:
        key = comm.comm_key
        nodes[key] = NodeEmbedding(
            video_id=enhanced_kg.video_id,
            node_type="community",
            label=comm.title,
            entity_only_embedding=None,
            entity_event_embedding=None,
            full_heterogeneous_embedding=emb_C.get(key),
        )

    meta = Node2VecMeta(
        dim=dim,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        window=window,
        graphs=["entity_only", "entity_micro_event", "full_heterogeneous"],
        edge_weight_schema={
            "entity_entity": "co-occurrence count (summed)",
            "entity_micro_event": "1.0 (membership)",
            "NEXT_MICRO_EVENT": "2.0 (fixed)",
            "SEMANTICALLY_SIMILAR_MICRO": "1.5 * similarity",
            "SHARES_CONTEXT_MICRO": "1.2 * similarity",
            "CAUSAL_MICRO": "1.8 * similarity",
            "micro_event_big_event": "1.5 (parent link)",
            "event_entity": "1.0 (membership)",
            "NEXT_EVENT": "2.0 (fixed)",
            "SEMANTICALLY_SIMILAR": "1.5 * similarity",
            "SHARES_CONTEXT": "1.2 * similarity",
            "CAUSAL": "1.8 * similarity",
            "entity_community": "1.0 (membership)",
            "event_community": "max(1.0, shared_entities)",
        },
    )

    return Node2VecOutput(video_id=enhanced_kg.video_id, meta=meta, nodes=nodes)


def run_node2vec(
    enhanced_kg: EnhancedKG,
    communities: CommunitiesOutput,
    dim: int = 128,
    walk_length: int = 80,
    num_walks: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    window: int = 10,
    workers: int = 4,
    seed: int = 42,
) -> Node2VecOutput:
    """Run the full Node2Vec pipeline."""
    print(f"\n[Stage 5] Node2Vec Structural Embeddings")
    print(f"  entities={len(enhanced_kg.entities)}  events={len(enhanced_kg.events)}  communities={len(communities.communities)}")

    # Build graphs
    print(f"  Building graphs...")
    G_A = build_entity_only_graph(enhanced_kg)
    G_B = build_entity_micro_event_graph(enhanced_kg)
    G_C = build_full_heterogeneous_graph(enhanced_kg, communities)

    n2v_kwargs = dict(
        dim=dim, walk_length=walk_length, num_walks=num_walks,
        p=p, q=q, window=window, workers=workers, seed=seed,
    )

    # Train on each graph
    print(f"  Training node2vec on Graph A (entity_only)...")
    emb_A = train_node2vec(G_A, "entity_only", **n2v_kwargs) #type:ignore

    print(f"  Training node2vec on Graph B (entity_micro_event)...")
    emb_B = train_node2vec(G_B, "entity_micro_event", **n2v_kwargs) #type:ignore

    print(f"  Training node2vec on Graph C (full_heterogeneous)...")
    emb_C = train_node2vec(G_C, "full_heterogeneous", **n2v_kwargs) #type:ignore

    # Assemble output
    print(f"  Assembling output...")
    output = assemble_output(
        enhanced_kg, communities, emb_A, emb_B, emb_C,
        dim, walk_length, num_walks, p, q, window,
    )

    n_entities = sum(1 for v in output.nodes.values() if v.node_type == "entity")
    n_micro_events = sum(1 for v in output.nodes.values() if v.node_type == "micro_event")
    n_events = sum(1 for v in output.nodes.values() if v.node_type == "event")
    n_communities = sum(1 for v in output.nodes.values() if v.node_type == "community")

    print(f"  total nodes   : {len(output.nodes)}")
    print(f"    entities    : {n_entities}  (graphs A, B, C)")
    print(f"    micro-events: {n_micro_events}  (graphs B, C)")
    print(f"    events      : {n_events}  (graphs B, C)")
    print(f"    communities : {n_communities}  (graph C only)")
    print(f"  embedding dim : {dim}")

    return output