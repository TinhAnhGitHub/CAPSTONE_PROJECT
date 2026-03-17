"""Prompt and structured output model for combined Image Caption + OCR task."""

from pydantic import BaseModel, Field


class ImageCaptionOCR(BaseModel):
    """Structured output for image caption and OCR extraction."""

    caption: str = Field(
        ...,
        description="A dense, factual paragraph describing the entire image scene (50-120 words). "
        "Include foreground/background elements, environmental style, lighting, and visible activity. "
        "Use neutral, descriptive language suitable for embedding and retrieval.",
    )
    ocr_texts: list[str] = Field(
        default_factory=list,
        description="List of all visible text extracted from the image in reading order (left-to-right, top-to-bottom). "
        "Include numbers, labels, captions, signs, and any embedded text. "
        "Return empty list if no text is visible.",
    )


CAPTION_OCR_PROMPT = """Analyze this image and provide:

1. **Dense Caption**: Write a single paragraph (50-120 words) describing the entire scene using factual, declarative language.
   - Include foreground and background elements
   - Describe architectural or environmental style
   - Note lighting source, direction, and quality
   - Identify the primary visible activity or event
   - Avoid speculation, emotions, or inferred intent
   - Use neutral, descriptive phrasing suitable for retrieval

2. **OCR Text Extraction**: Extract all visible text from the image.
   - Transcribe text exactly as it appears
   - Maintain reading order (left-to-right, top-to-bottom)
   - Include numbers, labels, captions, signs, and embedded text
   - If no text is visible, return an empty list

Return a structured JSON with:
- `caption`: string (the dense description)
- `ocr_texts`: list of strings (extracted text items, or empty list if none)
"""