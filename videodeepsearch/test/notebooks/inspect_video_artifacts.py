#!/usr/bin/env python3
"""
Inspect video artifacts from ArangoDB (KG data) for all failed trace video IDs.
"""
from arango import ArangoClient
import json

VIDEO_IDS = [
    "69d3bac406491d0fe176745a",
    "69d392f406491d0fe1767449",
    "69d392f406491d0fe176744a",
    "69d3cc4c06491d0fe1767461",
    "69d3d8b506491d0fe1767468",
    "69d39c6006491d0fe176744f",
    "69d3dd1e06491d0fe176746d",
]

ARANGO_CONFIG = {
    "host": "http://localhost:8529",
    "database": "video_kg",
    "username": "root",
    "password": "",
}

def inspect_video(db, video_id):
    """Inspect all KG data for a single video."""
    print(f"\n{'='*80}")
    print(f"VIDEO: {video_id}")
    print("="*80)
    
    # Query video metadata from videos collection
    try:
        video = db.aql.execute(
            f"FOR doc IN videos FILTER doc._key == '{video_id}' RETURN doc"
        )
        video_list = list(video)
        if video_list:
            v = video_list[0]
            print(f"\n  VIDEO METADATA:")
            for key in ['title', 'description', 'duration', 'user_id', 'created_at']:
                if key in v:
                    print(f"    {key}: {v.get(key, 'N/A')}")
        else:
            print(f"\n  VIDEO: Not found in videos collection")
    except Exception as e:
        print(f"\n  VIDEO query error: {e}")
    
    # Query entities
    try:
        entities = db.aql.execute(
            f"FOR doc IN entities FILTER doc.video_id == '{video_id}' RETURN {{entity_name: doc.entity_name, entity_type: doc.entity_type, desc: doc.desc}}"
        )
        entities_list = list(entities)
        print(f"\n  ENTITIES ({len(entities_list)} found):")
        for i, ent in enumerate(entities_list[:15]):
            desc = ent.get('desc', 'N/A')
            if desc:
                desc = desc[:100]
            print(f"    [{i+1}] {ent.get('entity_name', 'N/A')} ({ent.get('entity_type', 'N/A')})")
            if desc and desc != 'N/A':
                print(f"         Desc: {desc}")
    except Exception as e:
        print(f"\n  ENTITIES query error: {e}")
    
    # Query events (with captions)
    try:
        events = db.aql.execute(
            f"FOR doc IN events FILTER doc.video_id == '{video_id}' SORT doc.segment_index RETURN {{segment_index: doc.segment_index, start_time: doc.start_time, end_time: doc.end_time, caption: doc.caption}}"
        )
        events_list = list(events)
        print(f"\n  EVENTS ({len(events_list)} found):")
        for i, ev in enumerate(events_list[:10]):
            caption = ev.get('caption', 'N/A')
            if caption:
                caption = caption[:150]
            print(f"    [{i+1}] Segment {ev.get('segment_index', 'N/A')}: {ev.get('start_time', 'N/A')} - {ev.get('end_time', 'N/A')}")
            print(f"         Caption: {caption}")
    except Exception as e:
        print(f"\n  EVENTS query error: {e}")
    
    # Query micro_events (with text/ASR)
    try:
        micro_events = db.aql.execute(
            f"FOR doc IN micro_events FILTER doc.video_id == '{video_id}' SORT doc.segment_index, doc.micro_index RETURN {{segment_index: doc.segment_index, micro_index: doc.micro_index, start_time: doc.start_time, end_time: doc.end_time, text: doc.text, related_caption_context: doc.related_caption_context}}"
        )
        micro_list = list(micro_events)
        print(f"\n  MICRO_EVENTS ({len(micro_list)} found):")
        for i, me in enumerate(micro_list[:10]):
            text = me.get('text', 'N/A')
            if text:
                text = text[:150]
            caption_ctx = me.get('related_caption_context', 'N/A')
            if caption_ctx:
                caption_ctx = caption_ctx[:100]
            print(f"    [{i+1}] Micro {me.get('micro_index', 'N/A')} @ Segment {me.get('segment_index', 'N/A')}: {me.get('start_time', 'N/A')} - {me.get('end_time', 'N/A')}")
            print(f"         Text (ASR): {text}")
            if caption_ctx and caption_ctx != 'N/A':
                print(f"         Caption Context: {caption_ctx}")
    except Exception as e:
        print(f"\n  MICRO_EVENTS query error: {e}")
    
    # Query communities
    try:
        communities = db.aql.execute(
            f"FOR doc IN communities FILTER doc.video_id == '{video_id}' RETURN {{title: doc.title, summary: doc.summary}}"
        )
        comm_list = list(communities)
        print(f"\n  COMMUNITIES ({len(comm_list)} found):")
        for i, comm in enumerate(comm_list[:5]):
            summary = comm.get('summary', 'N/A')
            if summary:
                summary = summary[:200]
            print(f"    [{i+1}] Title: {comm.get('title', 'N/A')}")
            print(f"         Summary: {summary}")
    except Exception as e:
        print(f"\n  COMMUNITIES query error: {e}")

def main():
    print("="*80)
    print("VIDEO ARTIFACT INSPECTION - FAILED TRACES")
    print("="*80)
    print(f"\nInspecting {len(VIDEO_IDS)} video IDs:")
    for vid in VIDEO_IDS:
        print(f"  - {vid}")
    
    client = ArangoClient(hosts=ARANGO_CONFIG["host"])
    db = client.db(ARANGO_CONFIG["database"], username=ARANGO_CONFIG["username"], password=ARANGO_CONFIG["password"])
    
    # Check collections
    collections = db.collections()
    print(f"\nAvailable ArangoDB collections: {[c['name'] for c in collections if not c['name'].startswith('_')]}")
    
    for video_id in VIDEO_IDS:
        inspect_video(db, video_id)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nKey observations:")
    print("  - Check if entities/events/micro_events contain ENGINEERING/MATERIAL content")
    print("  - Check if they contain WEATHER/TORNADO content instead")
    print("  - Verify actual content matches expected response claims")

if __name__ == "__main__":
    main()