from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class CaptionSegment(BaseModel):
    """Input from segment captioning stage.

    Represents one captioned audio segment with timing information.
    """
    video_id: str
    from_batch: int
    to_batch: int
    start_time: str 
    end_time: str
    start_sec: float
    end_sec: float
    summary_caption: str
    event_captions: list[str] = Field(default_factory=list)

class EntityDoc(BaseModel):
    """A single entity extracted from a video segment."""
    video_id: str
    entity_id: str
    entity_name: str
    entity_type: str
    desc: str
    vis_des: Optional[str] = None


class MicroEventDoc(BaseModel):
    """An atomic micro-event extracted from a video segment."""
    video_id: str
    event_id: str
    event_des: str


class RelationshipDoc(BaseModel):
    """A subject→object relationship within a segment."""
    video_id: str
    subject_id: str
    relation_desc: str
    object_id: str


class KGSegment(BaseModel):
    """Knowledge graph output for a single segment."""
    video_id: str
    from_batch: int
    to_batch: int
    start_time: str
    end_time: str
    start_sec: float
    end_sec: float
    summary_caption: str
    event_captions: list[str]
    entities: list[EntityDoc] = Field(default_factory=list)
    events: list[MicroEventDoc] = Field(default_factory=list)
    relationships: list[RelationshipDoc] = Field(default_factory=list)

class CanonicalEntity(BaseModel):
    """A globally canonical entity after resolution."""
    video_id: str
    global_entity_id: str
    entity_name: str
    entity_type: str
    desc: str
    semantic_embedding: list[float] = Field(default_factory=list)
    merged_from: list[str] = Field(default_factory=list)
    first_seen_segment: Optional[int] = None
    last_seen_segment: Optional[int] = None


class GlobalRelationship(BaseModel):
    """A collapsed, globally-remapped entity-entity relationship triple."""
    video_id: str
    subject_global: str
    relation_desc: str
    relation_desc_embedding: list[float] = Field(default_factory=list)
    object_global: str
    weight: int = 1
    seen_in_segments: list[int] = Field(default_factory=list)


class SegmentView(BaseModel):
    """Per-segment view of the resolved KG (useful for RAG retrieval)."""
    video_id: str
    segment_index: int
    from_batch: Optional[int] = None
    to_batch: Optional[int] = None
    start_time: str
    end_time: str
    start_sec: float
    end_sec: float
    caption: str
    entities: list[CanonicalEntity] = Field(default_factory=list)
    relationships: list[GlobalRelationship] = Field(default_factory=list)
    events: list[MicroEventDoc] = Field(default_factory=list)


class ResolvedKG(BaseModel):
    """Full resolved knowledge graph - output of Stage 2."""
    video_id: str
    entities: list[CanonicalEntity] = Field(default_factory=list)
    relationships: list[GlobalRelationship] = Field(default_factory=list)
    segments: list[SegmentView] = Field(default_factory=list)

class EventNode(BaseModel):
    """A segment-level event node in the knowledge graph."""
    video_id: str
    key: str
    segment_index: int
    start_time: str
    end_time: str
    start_sec: float
    end_sec: float
    caption: str
    entities_global: list[str] = Field(default_factory=list)
    micro_events: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = None

    def to_arango_doc(self) -> dict:
        d = self.model_dump(exclude={"key"})
        d["_key"] = self.key
        return d

    def to_raw_dict(self) -> dict:
        return self.to_arango_doc()


class EventEntityEdge(BaseModel):
    """Edge connecting an event node to an entity node."""
    video_id: str
    from_key: str
    to_key: str

    def to_arango_doc(self) -> dict:
        return {"_from": self.from_key, "_to": self.to_key}


class EventEdge(BaseModel):
    """Edge connecting two event nodes with a typed relationship."""
    video_id: str
    from_key: str
    to_key: str
    edge_type: str
    temporal_gap_s: str
    similarity: Optional[float] = None
    shared_entities: int = 0
    jaccard: Optional[float] = None
    llm_reason: Optional[str] = None
    llm_confirmed: bool = False

    def to_arango_doc(self) -> dict:
        d = self.model_dump(exclude={"from_key", "to_key"})
        d["_from"] = self.from_key
        d["_to"] = self.to_key
        return d


class MicroEventNode(BaseModel):
    """A micro-event node in the knowledge graph."""
    video_id: str
    key: str
    parent_event_key: str
    segment_index: int
    micro_index: int
    start_time: str
    related_caption_context: str
    end_time: str
    start_secs: float
    end_secs: float
    text: str
    entities_global: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None

    def to_arango_doc(self) -> dict:
        d = self.model_dump(exclude={"key"})
        d["_key"] = self.key
        return d

    def to_raw_dict(self) -> dict:
        return self.to_arango_doc()


class MicroEventEdge(BaseModel):
    """Edge connecting two micro-event nodes."""
    video_id: str
    from_key: str
    to_key: str
    edge_type: str
    temporal_gap_s: str
    similarity: Optional[float] = None
    shared_entities: int = 0
    jaccard: Optional[float] = None
    llm_reason: Optional[str] = None
    llm_confirmed: bool = False

    def to_arango_doc(self) -> dict:
        d = self.model_dump(exclude={"from_key", "to_key"})
        d["_from"] = self.from_key
        d["_to"] = self.to_key
        return d


class EnhancedKG(BaseModel):
    """Resolved KG extended with the event layer - output of Stage 3."""
    video_id: str

    entities: list[CanonicalEntity] = Field(default_factory=list)
    relationships: list[GlobalRelationship] = Field(default_factory=list)
    segments: list[SegmentView] = Field(default_factory=list)

    events: list[EventNode] = Field(default_factory=list)
    event_entity_links: list[EventEntityEdge] = Field(default_factory=list)
    event_edges: list[EventEdge] = Field(default_factory=list)

    micro_event_nodes: list[MicroEventNode] = Field(default_factory=list)
    micro_event_edges: list[MicroEventEdge] = Field(default_factory=list)

    def to_raw_dict(self) -> dict:
        """Convert to dict format for downstream processing."""
        d = self.model_dump(
            exclude={"events", "event_entity_links", "event_edges",
                     "micro_event_nodes", "micro_event_edges"}
        )
        d["events"] = [ev.to_raw_dict() for ev in self.events]
        d["event_entity_links"] = [e.to_arango_doc() for e in self.event_entity_links]
        d["event_edges"] = [e.to_arango_doc() for e in self.event_edges]
        d["micro_event_nodes"] = [mn.to_raw_dict() for mn in self.micro_event_nodes]
        d["micro_event_edges"] = [me.to_arango_doc() for me in self.micro_event_edges]
        return d

class CommunityDoc(BaseModel):
    """One Leiden community of tightly connected entities."""
    video_id: str
    comm_key: str
    comm_idx: int
    title: str
    summary: str
    size: int
    member_keys: list[str] = Field(default_factory=list)
    event_keys: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = None


class MembershipEdge(BaseModel):
    """Edge: entity → community."""
    video_id: str
    from_key: str
    to_key: str

    def to_arango_doc(self) -> dict:
        return {"_from": self.from_key, "_to": self.to_key}


class EventCommunityEdge(BaseModel):
    """Edge: event → community (assigned by majority vote on shared entities)."""
    video_id: str
    from_key: str
    to_key: str
    shared_entities: int = 0
    assignment: str = "majority_vote"

    def to_arango_doc(self) -> dict:
        d = self.model_dump(exclude={"from_key", "to_key"})
        d["_from"] = self.from_key
        d["_to"] = self.to_key
        return d


class GraphStats(BaseModel):
    """Statistics from the Leiden community detection run."""
    n_nodes: int
    n_edges: int
    n_communities: int
    modularity: float


class CommunitiesOutput(BaseModel):
    """Full output of Stage 4 (community detection + LLM summaries)."""
    video_id: str
    communities: list[CommunityDoc] = Field(default_factory=list)
    membership_edges: list[MembershipEdge] = Field(default_factory=list)
    event_community_edges: list[EventCommunityEdge] = Field(default_factory=list)
    graph_stats: GraphStats

    def to_raw_dict(self) -> dict:
        """Dict format for downstream processing."""
        return {
            "communities": [c.model_dump() for c in self.communities],
            "membership_edges": [e.to_arango_doc() for e in self.membership_edges],
            "event_community_edges": [e.to_arango_doc() for e in self.event_community_edges],
            "graph_stats": self.graph_stats.model_dump(),
        }
        
class Node2VecMeta(BaseModel):
    """Hyperparameters used when training the node2vec models."""
    dim: int
    walk_length: int
    num_walks: int
    p: float
    q: float
    window: int
    graphs: list[str] = Field(default_factory=list)
    edge_weight_schema: dict[str, str] = Field(default_factory=dict)


class NodeEmbedding(BaseModel):
    """Structural node2vec embeddings for a single graph node."""
    video_id: str
    node_type: str  # "entity" | "event" | "community"
    label: str
    entity_only_embedding: Optional[list[float]] = None
    entity_event_embedding: Optional[list[float]] = None
    full_heterogeneous_embedding: Optional[list[float]] = None


class Node2VecOutput(BaseModel):
    """Full output of Stage 5 - structural embeddings for all node types."""
    video_id: str
    meta: Node2VecMeta
    nodes: dict[str, NodeEmbedding] = Field(default_factory=dict)

class VideoPipelineKGResult(BaseModel):
    """Result of processing one video through all KG pipeline stages."""
    video_id: str
    enhanced_kg: Optional[EnhancedKG] = None
    communities: Optional[CommunitiesOutput] = None
    node2vec: Optional[Node2VecOutput] = None
    error: Optional[str] = None

class CostTracker(BaseModel):
    """Track LLM costs across all KG pipeline stages."""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0
    llm_calls: int = 0
    model: str = ""

    def add_usage(self, prompt_tokens: int, completion_tokens: int, cost: float = 0.0) -> None:
        """Add usage from a single LLM call."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += cost
        self.llm_calls += 1

    def merge(self, other: "CostTracker") -> None:
        """Merge another CostTracker into this one."""
        self.total_prompt_tokens += other.total_prompt_tokens
        self.total_completion_tokens += other.total_completion_tokens
        self.total_cost += other.total_cost
        self.llm_calls += other.llm_calls