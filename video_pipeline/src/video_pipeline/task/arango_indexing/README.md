# ArangoDB Indexing Task

## Purpose

`arango_indexing_task` writes the finalized `KGGraphArtifact` into ArangoDB collections for graph retrieval. It creates vertex documents for videos, entities, events, and micro-events, then creates edge documents for relationships and temporal/context links.

## How It Works

The task implementation is `ArangoIndexingTask` in `main.py`.

1. `preprocess()` validates and returns the `KGGraphArtifact`.
2. `execute()` connects to ArangoDB and sets up the configured database/graph.
3. `execute()` inserts or replaces one `videos` document for the source video.
4. `execute()` embeds entity descriptions, event captions, and micro-event text with mmBERT.
5. `execute()` inserts `entities`, `events`, and `micro_events` vertex documents.
6. `execute()` inserts edge collections for entity relations, event sequences, event-entity links, micro-event sequences, micro-event parents, and micro-event-entity links.
7. `execute()` returns an `ArangoIndexingArtifact` containing insertion counts.
8. `summary_artifact()` writes ArangoDB collection counts to Prefect.

## Input

Type: `KGGraphArtifact`

| Field | Meaning |
|-------|---------|
| `entities` | Canonical entity vertices. |
| `relationships` | Entity-to-entity edges. |
| `events` | Segment-level event vertices. |
| `event_entity_links` | Event-to-entity edges. |
| `event_edges` | Event-to-event edges. |
| `micro_event_nodes` | Micro-event vertices. |
| `micro_event_edges` | Micro-event-to-micro-event edges. |

## Output

Type: `ArangoIndexingArtifact`

| Field | Meaning |
|-------|---------|
| `related_kg_artifact_id` | Source KG artifact ID. |
| `entities` | Number of entity documents inserted. |
| `events` | Number of event documents inserted. |
| `micro_events` | Number of micro-event documents inserted. |
| `entity_relations` | Number of entity relation edges inserted. |
| `event_sequences` | Number of event-event edges inserted. |
| `event_entities` | Number of event-entity edges inserted. |
| `micro_event_sequences` | Number of micro-event edges inserted. |
| `micro_event_parents` | Number of micro-event parent edges inserted. |
| `micro_event_entities` | Number of micro-event-entity edges inserted. |

## Dask Parallelization

The flow submits this task once with `arango_indexing_task.submit(kg_artifact, wait_for=kg_futures)`. It runs on a Dask worker but is not mapped because the graph must be indexed as a coherent video-level unit.

## Graph Schema

Vertex collections:

| Collection | Contents |
|------------|----------|
| `videos` | One document per processed video. |
| `entities` | Canonical entities with semantic embeddings. |
| `events` | Segment-level event nodes with semantic embeddings. |
| `micro_events` | Fine-grained micro-event nodes with semantic embeddings. |

Edge collections:

| Collection | Edge |
|------------|------|
| `entity_relations` | `entities -> entities` relationship triples. |
| `event_sequences` | `events -> events` temporal, semantic, context, or causal links. |
| `event_entities` | `events -> entities` participation links. |
| `micro_event_sequences` | `micro_events -> micro_events` micro-level links. |
| `micro_event_parents` | `micro_events -> events` hierarchy links. |
| `micro_event_entities` | `micro_events -> entities` participation links. |

## Algorithm Details

All Arango document keys are namespaced as `{video_id}::{local_key}` to avoid collisions across videos. Incoming edge endpoints from `KGGraphArtifact` use collection-prefixed IDs such as `events/event_0001`; `_strip_collection()` removes the original collection prefix, and `_ns()` applies the video namespace before insertion.

Vector indexes are not created by this task. The module comments state that ArangoDB IVF indexes should be created separately after data is loaded because they need training data.
