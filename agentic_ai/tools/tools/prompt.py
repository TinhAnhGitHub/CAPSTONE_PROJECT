CAPTION = """
write concise, descriptive captions for the image, remember:
- Start the caption with: "The pictures shows ... (brief scenery description)"
- Focus on the main subjects and actions in the image using this grammar: S + V + O.
- Use clear and specific language. If can, break down complex scenes into multiple sentences.
- Each entity should be distinguishable. If an entity is too general, add more details.

Example captions:
The picture shows a birthday party. The woman holds a chocolate birthdaycake. There are 8 candels on the cake.
"""

VISUAL = """
Given an image, identify MAXIMUM 5 most distinct entities (people, objects, animals, etc.) present in the scene.
Please return only a JSON object strictly formatted as such:

{
  "entities": [
    {
      "type": "string" // label e.g., 'human', 'ocr', 'car',
      "id": "string" // 10 letters unique identifier, e.g., 'a1e0345', 'bc567sd', ....
      "position": "string" // 'left', 'right', 'center', 'foreground', 'background', 
      "count": integer // for objects only, default is 1, only group > 1 if multiple identical objects,
      "description": "string" // ~15 words describe entity's attributes,
      "action": "string" // ~5 words describe entity's current actions, default is unmoved
    }
  ],
  "relationships": [
    {
      "subject_id": "string" // type:id of entity, eg: "person:01",
      "relation": "string (as concise and correct as possible. e.g., 'hold', 'next to')",
      "target_id": "string" /// type:id of entity, eg: "person:03"
    }  ]
}

Remember: 
- Return valid JSON only, no extra text, no syntax error (close paranthesis, closig quote, ......).
- Return exact fields with data types (id must be int)
- include all clearly visible entities (OCR, objects). For type == "ocr", attributes are always only colour and text.
- Humans must include `sex`, `clothes`, and `action` in description (if visible).
"""

QA = """
"""


REL = """
Given the image, a json contain list of entities = {l} in a scene, return a JSON strictly formatted as such:
'''json
{{
  "relationships": [
    {{
      "subject_id": "string" // id,
      "relation": "string (as concise and correct as possible. e.g., 'hold', 'next to')",
      "target_id": "string" // id
    }}
  ]
}}
'''
"""

"""
Remember to:
- Return valid JSON with exact amount of attributes, no extra text, no syntax error.

Example output with L = ["person:01", "person:02", "person:03"]:
{{
  "relationships": [
    {{
      "subject": "person:01", (must match exactly an entity in L)
      "relation": "punch",
      "target": "person:03" (must match exactly an entity in L)
    }}
  ]
}}
"""
MOONDREAM_PROMPT = {
    "caption": CAPTION,
    "visual": VISUAL,
    "query": QA,
    "relationship": REL
}

fix_detect = """
    You are a JSON repair assistant.

    The following JSON string could not be parsed due to the error shown below.
    Your task is to fix it and output only **valid JSON**, nothing else.

    ---
    ❌ JSON Error:
    {error_msg}

    🧩 Malformed JSON:
    '''json
    {broken_json}
    '''
    ✅ Repaired JSON (output only valid JSON):
    """

fix_rel = f"""
    You are a JSON repair assistant.  

"""