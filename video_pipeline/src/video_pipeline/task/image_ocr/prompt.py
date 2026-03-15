"""OCR prompt for extracting text from images."""

OCR_PROMPT = """Extract all visible text from this image.

Instructions:
- Transcribe all text exactly as it appears
- Maintain the reading order (left-to-right, top-to-bottom)
- Include numbers, labels, captions, and any embedded text
- If the image contains no text, respond with an empty string
- Do not add any commentary or explanation

Output only the extracted text, nothing else. If there is not scannable text on the picture, just return empty string.
"""