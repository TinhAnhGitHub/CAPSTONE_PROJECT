from llama_index.core.prompts import PromptTemplate

CONTRASTIVE_VISUAL_ENHANCEMENT_PROMPT = PromptTemplate(
    """
    You are an expert in CLIP prompt engineering and multimodal retrieval.
    Your goal is to create contrastively effective visual prompts that maximize alignment 
    between text and image embeddings.

    Guidelines:
    - Use natural language templates that match CLIP's training distribution 
      (e.g., “a photo of…”, “a painting of…”, “a detailed picture of…”).
    - Include relevant visual modifiers such as lighting, color, texture, composition, 
      and emotional tone.
    - Add descriptive context: background, setting, and object relationships.
    - Generate *multiple* prompt variations (3-5) that capture distinct but semantically 
      consistent views of the same concept, if asked
    - Avoid domain-unfamiliar jargon unless explicitly required.
    - Make sure each prompt is under 25 words and coherent.

    Input: {raw_query}
    variants: {variants}

    Output: A list of optimized CLIP-style prompts suitable for contrastive retrieval.
    """
)

CAPTION__ENHANCEMENT_PROMPT = PromptTemplate(
    """
    You are a visual captioning expert. 
    Generate a single, detailed sentence that describes the given image vividly 
    but concisely for semantic embedding retrieval.

    Guidelines:
    - Focus on *what is distinctive* in the image: objects, relationships, actions, context.
    - Include *visual attributes* (color, shape, texture, mood).
    - Prefer objective description over subjective opinion.
    - Avoid redundant phrases like "This image shows...".
    - Output one sentence only.

    Input: {raw_query}
    variants: {variants
    Output: A list of retrieval-optimized caption.
    """
)