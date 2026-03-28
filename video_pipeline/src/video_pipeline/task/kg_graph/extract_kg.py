from __future__ import annotations

import asyncio
import uuid

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from .models import CaptionSegment, KGSegment, EntityDoc, MicroEventDoc, RelationshipDoc, CostTracker
from .prompt import GRAPH_PROMPT



class Entity(BaseModel):
    entity_id: str = Field(..., description="Unique Entity ID (e.g., entity_01)")
    entity_name: str = Field(..., description="Unique Name (e.g., Man, Red Fridge)")
    entity_type: str = Field(..., description="The entity type (like Person, Object, Location)")
    desc: str = Field(..., description="General description from the context.")
    vis_des: str | None = Field(None, description="Visual description if available.")


class Relationship(BaseModel):
    subject_id: str = Field(..., description="The subject entity/event ID.")
    relation_desc: str = Field(..., description="The relationship description.")
    object_id: str = Field(..., description="The object entity/event ID.")


class Event(BaseModel):
    event_id: str = Field(..., description="The event id (i.e. event_01, event_02)")
    event_des: str = Field(..., description="The concise event description")


class VideoGraphExtraction(BaseModel):
    entities: list[Entity] = Field(..., description="list of all distinct entities.")
    relations: list[Relationship] = Field(..., description="List of relationships.")


def globalize_ids(segments: list[KGSegment]) -> list[KGSegment]:
    for segment in segments:
        entity_id_map: dict[str, str] = {}
        event_id_map: dict[str, str] = {}

        for entity in segment.entities:
            old_id = entity.entity_id
            new_id = f"entity_{uuid.uuid4()}"
            entity_id_map[old_id] = new_id
            entity.entity_id = new_id

        for event in segment.events:
            old_id = event.event_id
            new_id = f"event_{uuid.uuid4()}"
            event_id_map[old_id] = new_id
            event.event_id = new_id

        for rel in segment.relationships:
            if rel.subject_id in entity_id_map:
                rel.subject_id = entity_id_map[rel.subject_id]
            elif rel.subject_id in event_id_map:
                rel.subject_id = event_id_map[rel.subject_id]

            if rel.object_id in entity_id_map:
                rel.object_id = entity_id_map[rel.object_id]
            elif rel.object_id in event_id_map:
                rel.object_id = event_id_map[rel.object_id]

    return segments


def caption_segment_from_artifact(artifact) -> CaptionSegment:
    return CaptionSegment(
        video_id=artifact.related_video_id,
        from_batch=artifact.start_frame,
        to_batch=artifact.end_frame,
        start_time=artifact.start_timestamp,
        end_time=artifact.end_timestamp,
        start_sec=artifact.start_sec,
        end_sec=artifact.end_sec,
        summary_caption=artifact.summary_caption,
        event_captions=artifact.event_captions,
    )


def build_user_prompt(segment: CaptionSegment) -> str:
    events = segment.event_captions
    events_text = "\n".join([f"- {ev}" for ev in events]) if events else "No event captions."

    return f"""
    Extract the knowledge graph from the following video segment description.

    Segment [{segment.start_time} → {segment.end_time}]:
    Summary Caption: {segment.summary_caption}

    Event Captions:
    {events_text}
    """


async def extract_kg_from_segment(
    segment: CaptionSegment,
    as_llm,
    semaphore: asyncio.Semaphore,
    cost_tracker: CostTracker,
) -> KGSegment:
    events = segment.event_captions
    events_object = [
        Event(event_id=f"event_{i:02d}", event_des=ev)
        for i, ev in enumerate(events)
    ]

    user_prompt = build_user_prompt(segment)

    messages = [
        SystemMessage(content=GRAPH_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    async with semaphore:
        try:
            kg, usage = await as_llm(messages)

            prompt_tokens = usage.get('prompt_tokens', 0) or 0
            completion_tokens = usage.get('completion_tokens', 0) or 0
            cost = usage.get('cost', 0.0) or 0.0
            cost_tracker.add_usage(prompt_tokens, completion_tokens, cost)

            return KGSegment(
                video_id=segment.video_id,
                from_batch=segment.from_batch,
                to_batch=segment.to_batch,
                start_time=segment.start_time,
                end_time=segment.end_time,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                summary_caption=segment.summary_caption,
                event_captions=segment.event_captions,
                entities=[
                    EntityDoc(
                        video_id=segment.video_id,
                        entity_id=e.entity_id,
                        entity_name=e.entity_name,
                        entity_type=e.entity_type,
                        desc=e.desc,
                        vis_des=e.vis_des,
                    )
                    for e in kg.entities
                ],
                events=[
                    MicroEventDoc(
                        video_id=segment.video_id,
                        event_id=ev.event_id,
                        event_des=ev.event_des,
                    )
                    for ev in events_object
                ],
                relationships=[
                    RelationshipDoc(
                        video_id=segment.video_id,
                        subject_id=rel.subject_id,
                        relation_desc=rel.relation_desc,
                        object_id=rel.object_id,
                    )
                    for rel in kg.relations
                ],
            )

        except Exception as e:
            return KGSegment(
                video_id=segment.video_id,
                from_batch=segment.from_batch,
                to_batch=segment.to_batch,
                start_time=segment.start_time,
                end_time=segment.end_time,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                summary_caption=segment.summary_caption,
                event_captions=segment.event_captions,
            )


async def extract_kg_graph(
    captions: list[CaptionSegment],
    llm_client,
    max_concurrent: int = 5,
    cost_tracker: CostTracker | None = None,
) -> list[KGSegment]:
    as_llm = llm_client.as_structured_llm(VideoGraphExtraction)
    semaphore = asyncio.Semaphore(max_concurrent)

    if cost_tracker is None:
        cost_tracker = CostTracker()

    tasks = [
        extract_kg_from_segment(segment, as_llm, semaphore, cost_tracker)
        for segment in captions
    ]

    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: (x.from_batch, x.to_batch))
    results = globalize_ids(results)



    return results