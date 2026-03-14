PROMPT = """
Deconstruct this image into a structured JSON format optimized for:
- high-density vector retrieval

Follow the requirements exactly. The output must be factual, literal, and visually grounded.

Requirements:
Dense Caption (Global Semantic Anchor)
- Write a single paragraph of at least 100 words.
- Describe the entire scene using factual, declarative language.
- Include:
  - foreground and background elements
  - architectural or environmental style
  - lighting source, direction, and softness
  - the primary visible activity or event occurring in the image
- Avoid speculation, emotions, or inferred intent.
- Use neutral, descriptive phrasing suitable for embedding and retrieval.
- Around 50-120 words.

OUTPUT FORMAT:
A single clean caption with text.
```
"""