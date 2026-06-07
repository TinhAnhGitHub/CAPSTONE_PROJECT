#!/usr/bin/env python3
"""Migrate ArangoDB KG data from ``video_kg`` to ``video_kg_2``.

This script:
1. Creates the target database if it does not exist.
2. Creates the current collections and named graph.
3. Copies all documents from the source database.
4. Removes obsolete structural embedding fields while copying vertex docs.
5. Recreates only the current indexes used by the codebase.
6. Verifies migrated counts and obsolete-field cleanup.

Environment variables:
    ARANGO_HOST=http://localhost:8529
    ARANGO_USERNAME=root
    ARANGO_PASSWORD=
    SOURCE_DB=video_kg
    TARGET_DB=video_kg_2
"""

from __future__ import annotations

import math
import os
from typing import Any

from arango import ArangoClient
from arango.database import StandardDatabase


ARANGO_HOST = os.getenv("ARANGO_HOST", "http://localhost:8529")
ARANGO_USERNAME = os.getenv("ARANGO_USERNAME", "root")
ARANGO_PASSWORD = os.getenv("ARANGO_PASSWORD", "")

SOURCE_DB = os.getenv("SOURCE_DB", "video_kg")
TARGET_DB = os.getenv("TARGET_DB", "video_kg_2")
GRAPH_NAME = "video_knowledge_graph"
SEMANTIC_DIM = 768

VERTEX_COLLECTIONS = [
    "videos",
    "entities",
    "events",
    "micro_events",
]

EDGE_COLLECTIONS = [
    "entity_relations",
    "event_sequences",
    "event_entities",
    "micro_event_sequences",
    "micro_event_parents",
    "micro_event_entities",
]

ALL_COLLECTIONS = VERTEX_COLLECTIONS + EDGE_COLLECTIONS

OBSOLETE_VERTEX_FIELDS = {
    "structural_embedding_entity_only",
    "structural_embedding_entity_event",
    "structural_embedding_full",
}


def connect() -> tuple[ArangoClient, StandardDatabase, StandardDatabase]:
    client = ArangoClient(hosts=ARANGO_HOST)
    sys_db = client.db(
        "_system",
        username=ARANGO_USERNAME,
        password=ARANGO_PASSWORD,
    )

    if not sys_db.has_database(SOURCE_DB):
        raise RuntimeError(f"Source database does not exist: {SOURCE_DB}")

    if not sys_db.has_database(TARGET_DB):
        sys_db.create_database(TARGET_DB)
        print(f"[OK] Created database: {TARGET_DB}")
    else:
        print(f"[OK] Target database already exists: {TARGET_DB}")

    source = client.db(
        SOURCE_DB,
        username=ARANGO_USERNAME,
        password=ARANGO_PASSWORD,
    )
    target = client.db(
        TARGET_DB,
        username=ARANGO_USERNAME,
        password=ARANGO_PASSWORD,
    )
    return client, source, target


def ensure_collections(db: StandardDatabase) -> None:
    for name in VERTEX_COLLECTIONS:
        if not db.has_collection(name):
            db.create_collection(name)
            print(f"[OK] Created vertex collection: {name}")

    for name in EDGE_COLLECTIONS:
        if not db.has_collection(name):
            db.create_collection(name, edge=True)
            print(f"[OK] Created edge collection: {name}")


def ensure_graph(db: StandardDatabase) -> None:
    if db.has_graph(GRAPH_NAME):
        print(f"[OK] Graph already exists: {GRAPH_NAME}")
        return

    graph = db.create_graph(GRAPH_NAME)
    graph.create_edge_definition(
        edge_collection="entity_relations",
        from_vertex_collections=["entities"],
        to_vertex_collections=["entities"],
    )
    graph.create_edge_definition(
        edge_collection="event_sequences",
        from_vertex_collections=["events"],
        to_vertex_collections=["events"],
    )
    graph.create_edge_definition(
        edge_collection="event_entities",
        from_vertex_collections=["events"],
        to_vertex_collections=["entities"],
    )
    graph.create_edge_definition(
        edge_collection="micro_event_sequences",
        from_vertex_collections=["micro_events"],
        to_vertex_collections=["micro_events"],
    )
    graph.create_edge_definition(
        edge_collection="micro_event_parents",
        from_vertex_collections=["micro_events"],
        to_vertex_collections=["events"],
    )
    graph.create_edge_definition(
        edge_collection="micro_event_entities",
        from_vertex_collections=["micro_events"],
        to_vertex_collections=["entities"],
    )
    print(f"[OK] Created graph: {GRAPH_NAME}")


def sanitize_document(collection: str, doc: dict[str, Any]) -> dict[str, Any]:
    cleaned = {key: value for key, value in doc.items() if key not in {"_id", "_rev"}}

    if collection in {"entities", "events", "micro_events"}:
        for field in OBSOLETE_VERTEX_FIELDS:
            cleaned.pop(field, None)

    return cleaned


def copy_collection(
    source: StandardDatabase,
    target: StandardDatabase,
    collection: str,
    batch_size: int = 1000,
) -> int:
    dst_coll = target.collection(collection)
    cursor = source.aql.execute(
        f"FOR doc IN {collection} RETURN doc",
        batch_size=batch_size,
    )

    copied = 0
    batch: list[dict[str, Any]] = []

    for doc in cursor:
        batch.append(sanitize_document(collection, doc))
        if len(batch) >= batch_size:
            dst_coll.insert_many(
                batch,
                overwrite_mode="replace",
                raise_on_document_error=True,
            )
            copied += len(batch)
            batch.clear()

    if batch:
        dst_coll.insert_many(
            batch,
            overwrite_mode="replace",
            raise_on_document_error=True,
        )
        copied += len(batch)

    print(f"[OK] Copied {copied} docs into {collection}")
    return copied


def safe_nlists(doc_count: int) -> int:
    if doc_count < 2:
        return 1
    return max(1, min(int(math.sqrt(doc_count)), doc_count))


def create_indexes(db: StandardDatabase) -> None:
    def add_index(collection: str, spec: dict[str, Any]) -> None:
        coll = db.collection(collection)
        existing = {idx.get("name") for idx in coll.indexes() if idx.get("name")}
        if spec["name"] in existing:
            print(f"[SKIP] Index already exists: {collection}.{spec['name']}")
            return
        coll.add_index(spec)
        print(f"[OK] Created index: {collection}.{spec['name']}")

    semantic_targets = [
        ("entities", "entity_semantic_idx", ["entity_type", "global_entity_id"]),
        ("events", "event_semantic_idx", ["segment_index", "start_time"]),
        ("micro_events", "micro_event_semantic_idx", ["segment_index", "micro_index"]),
    ]

    for collection, name, stored_values in semantic_targets:
        n_docs = db.collection(collection).count()
        n_lists = safe_nlists(n_docs)
        add_index(
            collection,
            {
                "type": "vector",
                "name": name,
                "fields": ["semantic_embedding"],
                "params": {
                    "dimension": SEMANTIC_DIM,
                    "metric": "cosine",
                    "nLists": n_lists,
                    "defaultNProbe": min(10, n_lists),
                    "trainingIterations": 40,
                },
                "storedValues": stored_values,
            },
        )

    add_index(
        "entities",
        {
            "type": "inverted",
            "name": "entities_inverted_idx",
            "fields": [
                {"name": "video_id"},
                {"name": "entity_type"},
                {"name": "global_entity_id"},
                {"name": "entity_name", "analyzer": "text_en"},
            ],
            "storedValues": [
                {"fields": ["desc"]},
                {"fields": ["first_seen_segment"]},
                {"fields": ["last_seen_segment"]},
            ],
            "optimizeTopK": ["BM25(@doc) DESC", "TFIDF(@doc) DESC"],
        },
    )
    add_index(
        "events",
        {
            "type": "inverted",
            "name": "events_inverted_idx",
            "fields": [
                {"name": "video_id"},
                {"name": "segment_index"},
                {"name": "caption", "analyzer": "text_en"},
            ],
            "storedValues": [
                {"fields": ["start_time"]},
                {"fields": ["end_time"]},
                {"fields": ["start_sec"]},
                {"fields": ["end_sec"]},
            ],
            "optimizeTopK": ["BM25(@doc) DESC", "TFIDF(@doc) DESC"],
        },
    )
    add_index(
        "micro_events",
        {
            "type": "inverted",
            "name": "micro_events_inverted_idx",
            "fields": [
                {"name": "video_id"},
                {"name": "segment_index"},
                {"name": "parent_event_key"},
                {"name": "text", "analyzer": "text_en"},
                {"name": "related_caption_context", "analyzer": "text_en"},
            ],
            "storedValues": [
                {"fields": ["start_time"]},
                {"fields": ["end_time"]},
                {"fields": ["start_secs"]},
                {"fields": ["end_secs"]},
                {"fields": ["micro_index"]},
            ],
            "optimizeTopK": ["BM25(@doc) DESC", "TFIDF(@doc) DESC"],
        },
    )

    add_index(
        "events",
        {
            "type": "mdi",
            "name": "events_time_mdi_idx",
            "fields": ["start_sec", "end_sec"],
            "fieldValueTypes": "double",
        },
    )
    add_index(
        "micro_events",
        {
            "type": "mdi",
            "name": "micro_events_time_mdi_idx",
            "fields": ["start_secs", "end_secs"],
            "fieldValueTypes": "double",
        },
    )

    add_index(
        "entities",
        {
            "type": "persistent",
            "name": "entities_video_id_idx",
            "fields": ["video_id"],
            "sparse": True,
        },
    )
    add_index(
        "events",
        {
            "type": "persistent",
            "name": "events_video_segment_idx",
            "fields": ["video_id", "segment_index"],
            "sparse": True,
        },
    )
    add_index(
        "micro_events",
        {
            "type": "persistent",
            "name": "micro_events_video_segment_idx",
            "fields": ["video_id", "segment_index", "micro_index"],
            "sparse": True,
        },
    )


def verify_counts(source: StandardDatabase, target: StandardDatabase) -> bool:
    print("\n=== Count verification ===")
    ok = True

    for collection in ALL_COLLECTIONS:
        source_count = source.collection(collection).count()
        target_count = target.collection(collection).count()
        status = "OK" if source_count == target_count else "MISMATCH"
        print(
            f"{collection:28s} "
            f"source={source_count:<8d} "
            f"target={target_count:<8d} "
            f"{status}"
        )
        ok = ok and source_count == target_count

    return ok


def verify_obsolete_fields_removed(target: StandardDatabase) -> bool:
    print("\n=== Obsolete field verification ===")
    ok = True

    for collection in ("entities", "events", "micro_events"):
        query = f"""
        FOR d IN {collection}
            FILTER HAS(d, "structural_embedding_entity_only")
                OR HAS(d, "structural_embedding_entity_event")
                OR HAS(d, "structural_embedding_full")
            COLLECT WITH COUNT INTO n
            RETURN n
        """
        count = list(target.aql.execute(query))[0]
        print(f"{collection:28s} obsolete_field_docs={count}")
        ok = ok and count == 0

    return ok


def verify_no_obsolete_indexes(target: StandardDatabase) -> bool:
    print("\n=== Obsolete index verification ===")
    ok = True

    for collection in ("entities", "events", "micro_events"):
        index_names = {
            idx.get("name")
            for idx in target.collection(collection).indexes()
            if idx.get("name")
        }
        stale = sorted(
            name
            for name in index_names
            if name and name.startswith("structural_embedding")
        )
        print(f"{collection:28s} stale_indexes={stale}")
        ok = ok and not stale

    return ok


def main() -> int:
    client, source, target = connect()

    try:
        print("\n=== Creating target schema ===")
        ensure_collections(target)
        ensure_graph(target)

        print("\n=== Copying data ===")
        for collection in ALL_COLLECTIONS:
            copy_collection(source, target, collection)

        print("\n=== Creating current indexes ===")
        create_indexes(target)

        counts_ok = verify_counts(source, target)
        fields_ok = verify_obsolete_fields_removed(target)
        indexes_ok = verify_no_obsolete_indexes(target)

        print("\n=== Migration result ===")
        if counts_ok and fields_ok and indexes_ok:
            print("[SUCCESS] Migration completed and verified.")
            print(f"Source DB: {SOURCE_DB}")
            print(f"Target DB: {TARGET_DB}")
            return 0

        print("[FAILED] Migration completed but verification failed.")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
