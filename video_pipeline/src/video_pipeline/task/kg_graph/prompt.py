"""LLM prompts for Knowledge Graph extraction."""

GRAPH_PROMPT = """
You are a deterministic Knowledge Graph extraction engine.

Your task is to convert a single video segment description into a structured knowledge graph
containing:

- Entities
- Chronologically ordered micro-events
- Explicit relationships

You MUST follow the rules strictly.

-----------------------------------
GLOBAL RULES
-----------------------------------

1. Work ONLY with information explicitly present in the text.
2. Do NOT invent unseen objects or people.
3. Resolve ALL pronouns (he, she, it, they, the man, the woman, etc.)
   to their explicit canonical entity names.
4. If two mentions refer to the same real-world entity, merge them.
5. Every subject_id and object_id in relations MUST match an entity_id or event_id exactly.
6. Events MUST be chronological.
7. No duplicate entities.
8. No duplicate events.
9. Be concise but precise.

-----------------------------------
STEP 1 — ENTITY EXTRACTION
-----------------------------------

Extract ALL distinct entities mentioned in the segment.

For each entity provide:

- entity_id: entity_01, entity_02, ...
- entity_name: Canonical readable name (e.g., "Man", "Black Dog", "Wooden Table")
- entity_type: One of:
    - Person
    - Object
    - Animal
    - Location
    - Vehicle
    - Group
    - Other
- desc:
    General contextual description of the entity's role in this segment.
- vis_des:
    Only visible attributes (color, size, clothing, material, position, etc.)
    If no visual attributes are mentioned, output: "None"

Rules:
- Do NOT include abstract concepts unless clearly acting as an entity.
- Do NOT create separate entities for pronouns.
- If multiple similar objects exist, differentiate them clearly
  (e.g., "Man_1", "Man_2" only if truly different individuals).

-----------------------------------
STEP 2 — MICRO-EVENT SEGMENTATION
-----------------------------------

Break the segment into minimal chronological micro-events.

Each event must represent ONE atomic action.

GOOD:
- The man opens the door.
- The man enters the room.

BAD:
- The man opens the door and enters the room.  ← (split this)

For each event provide:

- event_id: event_01, event_02, ...
- event_des:
    A short, precise description of the single action.

-----------------------------------
CRITICAL CONSISTENCY RULE
-----------------------------------

Every:
- subject_id
- object_id

MUST exactly match one of:
- entity_id
- event_id

No free text allowed in IDs.

-----------------------------------
OUTPUT
-----------------------------------

Return ONLY structured output matching the schema.
Do not add explanations.
Do not add extra fields.
"""


def build_user_prompt(segment) -> str:
    """Build the user prompt for KG extraction from a segment."""
    from .models import CaptionSegment

    events = segment.event_captions
    events_text = "\n".join([f"- {ev}" for ev in events]) if events else "No event captions."

    return f"""
Extract the knowledge graph from the following video segment description.

Segment [{segment.start_time} → {segment.end_time}]:
Summary Caption: {segment.summary_caption}

Event Captions:
{events_text}
"""