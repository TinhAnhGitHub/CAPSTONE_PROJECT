"""Schema definitions for Knowledge Graph search results.

These schemas represent results returned from ArangoDB KG queries.
They follow the same pattern as ImageInterface and SegmentInterface.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KGEntityResult(BaseModel):
    """Entity search result from Knowledge Graph.

    Represents a canonical entity extracted from video content.
    """

    id: str = Field(..., description="Entity document key (namespaced)")
    video_id: str = Field(..., description="Video ID this entity belongs to")
    global_entity_id: str = Field(..., description="Global canonical entity ID")
    entity_name: str = Field(..., description="Name of the entity")
    entity_type: str = Field(..., description="Type of entity (PERSON, ORG, LOC, etc.)")
    desc: str = Field(default="", description="Entity description")
    score: float = Field(default=0.0, description="Search similarity score")
    rrf_score: float | None = Field(default=None, description="RRF fusion score if hybrid search")
    first_seen_segment: int | None = Field(default=None, description="First segment where entity appeared")
    last_seen_segment: int | None = Field(default=None, description="Last segment where entity appeared")
    merged_from: list[str] = Field(default_factory=list, description="Source entity IDs merged into this canonical")

    def brief_representation(self) -> str:
        """Return a brief string representation."""
        score_str = f"{self.score:.3f}" if self.score else "N/A"
        return f"[{self.entity_type}] {self.entity_name} (score={score_str})"

    def detailed_representation(self) -> str:
        """Return a detailed string representation."""
        lines = [
            f"Entity: {self.entity_name} [{self.entity_type}]",
            f"  ID: {self.global_entity_id}",
            f"  Video: {self.video_id}",
            f"  Score: {self.score:.3f}" if self.score else "  Score: N/A",
            f"  Description: {self.desc[:100]}..." if len(self.desc) > 100 else f"  Description: {self.desc}",
        ]
        if self.first_seen_segment is not None:
            lines.append(f"  Segments: {self.first_seen_segment} - {self.last_seen_segment}")
        return "\n".join(lines)


class KGEventResult(BaseModel):
    """Event search result from Knowledge Graph.

    Represents a segment-level event with caption and temporal info.
    """

    id: str = Field(..., description="Event document key (namespaced)")
    video_id: str = Field(..., description="Video ID this event belongs to")
    segment_index: int = Field(..., description="Segment index in video")
    caption: str = Field(..., description="Event caption/summary")
    start_time: str = Field(..., description="Start timestamp (HH:MM:SS.mmm)")
    end_time: str = Field(..., description="End timestamp (HH:MM:SS.mmm)")
    start_sec: float = Field(..., description="Start time in seconds")
    end_sec: float = Field(..., description="End time in seconds")
    score: float = Field(default=0.0, description="Search similarity score")
    entities_global: list[str] = Field(default_factory=list, description="Global entity IDs linked to this event")
    micro_events: list[str] = Field(default_factory=list, description="Micro-event keys within this event")

    def brief_representation(self) -> str:
        """Return a brief string representation."""
        return f"[{self.video_id}] Segment {self.segment_index}: {self.caption[:50]}... (score={self.score:.3f})"

    def detailed_representation(self) -> str:
        """Return a detailed string representation."""
        return (
            f"Event: {self.id}\n"
            f"  Video: {self.video_id} | Segment: {self.segment_index}\n"
            f"  Time: {self.start_time} - {self.end_time} ({self.start_sec:.1f}s - {self.end_sec:.1f}s)\n"
            f"  Caption: {self.caption}\n"
            f"  Score: {self.score:.3f}\n"
            f"  Entities: {len(self.entities_global)} | Micro-events: {len(self.micro_events)}"
        )


class KGMicroEventResult(BaseModel):
    """Micro-event search result from Knowledge Graph.

    Represents a fine-grained action/scene within an event.
    """

    id: str = Field(..., description="Micro-event document key (namespaced)")
    video_id: str = Field(..., description="Video ID this micro-event belongs to")
    parent_event_key: str = Field(..., description="Parent event key")
    segment_index: int = Field(..., description="Segment index in video")
    micro_index: int = Field(..., description="Micro-event index within segment")
    text: str = Field(..., description="Micro-event text description")
    related_caption_context: str = Field(default="", description="Related caption context")
    start_time: str = Field(..., description="Start timestamp")
    end_time: str = Field(..., description="End timestamp")
    start_secs: float | None = Field(default=None, description="Start time in seconds")
    end_secs: float | None = Field(default=None, description="End time in seconds")
    score: float = Field(default=0.0, description="Search similarity score")
    entities_global: list[str] = Field(default_factory=list, description="Linked entity IDs")

    def brief_representation(self) -> str:
        """Return a brief string representation."""
        return f"[{self.video_id}] Seg {self.segment_index}.{self.micro_index}: {self.text[:40]}... (score={self.score:.3f})"

    def detailed_representation(self) -> str:
        """Return a detailed string representation."""
        return (
            f"Micro-event: {self.id}\n"
            f"  Parent: {self.parent_event_key}\n"
            f"  Position: Segment {self.segment_index}.{self.micro_index}\n"
            f"  Time: {self.start_time} - {self.end_time}\n"
            f"  Text: {self.text}\n"
            f"  Score: {self.score:.3f}"
        )


class KGCommunityResult(BaseModel):
    """Community search result from Knowledge Graph.

    Represents a cluster of related entities forming a thematic group.
    """

    id: str = Field(..., description="Community document key (namespaced)")
    video_id: str = Field(..., description="Video ID this community belongs to")
    comm_idx: int = Field(..., description="Community index")
    title: str = Field(..., description="Community title")
    summary: str = Field(default="", description="Community summary")
    size: int = Field(default=0, description="Number of entities in community")
    score: float = Field(default=0.0, description="Search similarity score")
    member_keys: list[str] = Field(default_factory=list, description="Entity keys in this community")
    event_keys: list[str] = Field(default_factory=list, description="Event keys linked to this community")

    def brief_representation(self) -> str:
        """Return a brief string representation."""
        return f"[{self.video_id}] {self.title} (size={self.size}, score={self.score:.3f})"

    def detailed_representation(self) -> str:
        """Return a detailed string representation."""
        return (
            f"Community: {self.title}\n"
            f"  ID: {self.id} | Index: {self.comm_idx}\n"
            f"  Video: {self.video_id}\n"
            f"  Size: {self.size} entities\n"
            f"  Summary: {self.summary[:100]}..." if len(self.summary) > 100 else f"  Summary: {self.summary}\n"
            f"  Score: {self.score:.3f}"
        )


class KGTraversalResult(BaseModel):
    """Graph traversal result from Knowledge Graph.

    Represents a node reached via graph traversal with path info.
    """

    id: str = Field(..., description="Node document key")
    node_id: str = Field(..., description="Full ArangoDB document ID")
    video_id: str = Field(..., description="Video ID")
    label: str = Field(default="", description="Node label (entity name, caption, etc.)")
    edge_type: str = Field(default="", description="Type of edge traversed")
    weight: float | None = Field(default=None, description="Edge weight")
    path_length: int = Field(default=1, description="Length of traversal path")

    def brief_representation(self) -> str:
        """Return a brief string representation."""
        return f"[{self.edge_type}] -> {self.label} (depth={self.path_length})"

    def detailed_representation(self) -> str:
        """Return a detailed string representation."""
        weight_str = f" | weight={self.weight:.3f}" if self.weight else ""
        return (
            f"Node: {self.label}\n"
            f"  ID: {self.node_id}\n"
            f"  Video: {self.video_id}\n"
            f"  Edge: {self.edge_type}{weight_str}\n"
            f"  Path length: {self.path_length}"
        )


class KGRagResult(BaseModel):
    """Combined RAG retrieval result from Knowledge Graph.

    Contains entities, events, micro-events, communities, and graph context.
    """

    query: str = Field(..., description="Original query text")
    video_ids_searched: list[str] = Field(default_factory=list, description="Video IDs searched")
    entities: list[KGEntityResult] = Field(default_factory=list, description="Entity results")
    events: list[KGEventResult] = Field(default_factory=list, description="Event results")
    micro_events: list[KGMicroEventResult] = Field(default_factory=list, description="Micro-event results")
    communities: list[KGCommunityResult] = Field(default_factory=list, description="Community results")
    graph_context: list[KGTraversalResult] = Field(default_factory=list, description="Graph traversal context")
    videos_hit: dict[str, int] = Field(default_factory=dict, description="Video ID -> hit count")
    total_nodes: int = Field(default=0, description="Total number of nodes retrieved")

    def summary(self) -> str:
        """Return a summary of the RAG result."""
        return (
            f"RAG Result for: '{self.query}'\n"
            f"  Videos: {self.video_ids_searched}\n"
            f"  Entities: {len(self.entities)} | Events: {len(self.events)} | "
            f"Micro-events: {len(self.micro_events)} | Communities: {len(self.communities)}\n"
            f"  Graph context: {len(self.graph_context)} nodes\n"
            f"  Total: {self.total_nodes} nodes"
        )

    def to_context_string(self) -> str:
        """Convert result to a context string for LLM prompting."""
        sections = [f"=== Knowledge Graph Context ===\nQuery: {self.query}\n"]

        if self.entities:
            sections.append("\n--- Entities ---")
            for e in self.entities[:10]:
                sections.append(f"  - {e.brief_representation()}")

        if self.events:
            sections.append("\n--- Events ---")
            for ev in self.events[:5]:
                sections.append(f"  - {ev.brief_representation()}")

        if self.micro_events:
            sections.append("\n--- Micro-events ---")
            for me in self.micro_events[:5]:
                sections.append(f"  - {me.brief_representation()}")

        if self.communities:
            sections.append("\n--- Communities ---")
            for c in self.communities[:3]:
                sections.append(f"  - {c.brief_representation()}")

        if self.graph_context:
            sections.append("\n--- Graph Context ---")
            for g in self.graph_context[:10]:
                sections.append(f"  - {g.brief_representation()}")

        return "\n".join(sections)


class KGSearchStatistics(BaseModel):
    """Statistics about a KG search operation."""

    total_entities: int = Field(default=0, description="Total entities in result")
    total_events: int = Field(default=0, description="Total events in result")
    total_micro_events: int = Field(default=0, description="Total micro-events in result")
    total_communities: int = Field(default=0, description="Total communities in result")
    videos_represented: int = Field(default=0, description="Number of videos with hits")
    avg_score: float = Field(default=0.0, description="Average similarity score")
    search_methods: list[str] = Field(default_factory=list, description="Search methods used")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
