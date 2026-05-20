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

class VideoPipelineKGResult(BaseModel):
    """Result of processing one video through all KG pipeline stages."""
    video_id: str
    enhanced_kg: Optional[EnhancedKG] = None
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


class KGEntityResolutionResult(BaseModel):
    """Merged extraction output after global entity resolution."""

    resolved_kg: ResolvedKG
    cost_tracker: CostTracker
    user_id: str
    related_segment_caption_artifact_ids: list[str] = Field(default_factory=list)
    related_kg_extraction_artifact_ids: list[str] = Field(default_factory=list)
    total_raw_entities: int = 0
