"""
videodeepsearch/tools/implementation/util/str_template.py
"""

SNIPPET = """
    ASR transcript context around the segment:

    ▶ Segment range: {segment_start_time} → {segment_end_time}
    ▶ Frame range: {segment_start_frame} → {segment_end_frame}
    ▶ Context window: ±{window_seconds} seconds

    --------------------- TRANSCRIPT CONTEXT ---------------------
    {context}
    --------------------------------------------------------------

    Note: Some ASR lines may include adjacent context beyond the target segment.
    Focus on lines semantically aligned with the segment’s content.
    """


ASR_TOKEN_TEMPLATE = """
Start time/index: {token.start}/{token.start_frame}
End time/index:   {token.end}/{token.end_frame}
ASR content:      {token.text}
"""