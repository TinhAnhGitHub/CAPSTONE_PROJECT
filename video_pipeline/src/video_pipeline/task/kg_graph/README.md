# Kg Graph Task

## `extract_kg.py`
First we will investigate how we construct up the initial knowledge graph from just caption segments.
So for each video segment, we want to extract:
- Entities -> Objects, people, and places
- Events -> Actions happening
- Relationships -> how entities/events are connect

First, we will turn the events caption into Event Objects
```python
class Event(BaseModel):
    event_id: str = Field(..., description="The event id (i.e. event_01, event_02)")
    event_des: str = Field(..., description="The concise event description")
```

NExt, we willl build the prompt, where we will extract the VideoGraph Extraction from the event captions and summary caption

```python
class VideoGraphExtraction(BaseModel):
    entities: list[Entity] = Field(..., description="list of all distinct entities.")
    relations: list[Relationship] = Field(..., description="List of relationships.")
```

Then We call the LLM to do so, and then we will normalize the ID. Since each segment and entity use local id, we need to replace it with global unique IDS, and we also update inside the relationship.


## Stage 2: Entity Resolution

In this processing, we will resolce duplicate entities across segments into canonical (global) entities using hybrid similarity matching and LLM configuration

We first flatten entities from all KG segments into a flat list, then we will compute the similarity with hybrid embedding (dense and sparse) with a hybrid score. We use AgglomerativeClustering for the clustering computation. Next, for each cluster, we use LLM to confirm if entities refer to the same real-world entity. and create canonical entities. 

## Stage 3: Event Linking
For this processing, we build a hierarchical event layer with 2 levels:
- **BigEvents**: Segment-level events
- **MicroEvents**: Fine-grained events extracted from event caption
Creates edges between events based on semantic similarity, temporal proximity, and shared context

1. Build EventNode, for each segment with aggregated entities, dense embedding from caption, and EventEntityEdge
2. Using 4-pass algorithm to determine whether a pair of events are linked together.

The same applies to the MicroEvents




## Stage 4: Community Detection
DEtect communities of related entities using the Leiden algorithm and generate LLm summaries.

We create graph from igraph with entities + relationship, apply leiden algorithm. For each Community -> genetate llm summaries.

## Stage 5: Node2vec structural embeddings

Generate structural embeddings for all graph nodes using node2vec.
We will train on 3 kind of graph:
1. Entities only
2. Entities + micro events
3. Entities + events + micro-events + communities

## Storing into Arangodb

The whole knowledge graph , we will store them into the Arangodb with following edge collections:
```python
VERTEX_COLLECTIONS = [
    "videos",          # One doc per ingested video
    "entities",        # CanonicalEntity
    "events",          # EventNode (segment-level)
    "micro_events",    # MicroEventNode
    "communities",     # CommunityDoc
]

EDGE_COLLECTIONS = [
    "entity_relations",        # entity <-> entity
    "event_sequences",         # event  <-> event
    "event_entities",          # event  <-> entity
    "micro_event_sequences",   # micro_event <-> micro_event
    "micro_event_parents",     # micro_event -> event (parent)
    "micro_event_entities",    # micro_event <-> entity
    "community_members",       # entity -> community
    "event_communities",       # event  -> community
]
```

