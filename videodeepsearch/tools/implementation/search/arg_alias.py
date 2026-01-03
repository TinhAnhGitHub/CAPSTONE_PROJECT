"""
videodeepsearch/tools/implementation/search/arg_alias.py
"""
from typing import Annotated
from pydantic import Field

VisualQuery = Annotated[
    str,
    Field(
        description=(
            "A visually descriptive natural-language query. "
            "Avoid non-visual elements such as names, numbers, or abstract concepts. "
            "Must be in English."
        )
    )
]


TopKEach = Annotated[
    int, 
    "Number of top-matching images to retrieve based on caption embedding similarity."
]

TopkFinal = Annotated[
    int, "Number of top-matching images to retrieve based on caption embedding similarity."
]


DenseSparseWeight = Annotated[
    list[float] | None,
    "If provided, expects two weights [dense, sparse] for hybrid search."
]
MultiModalWeight = Annotated[
    list[float], "Expects three weights [visual, caption_dense, caption_sparse] for reranking."
]

TopK = Annotated[
    int,
    Field(
        description="Number of top-matching images to retrieve.",
        ge=1, 
        le=100, 
        default=50
    )
]

CaptionQuery =  Annotated[
    str,
    "A descriptive text query that semantically aligns with image captions."
    "Use this for retrieving images based on caption embeddings rather than raw visual content."
    "The caption query must be in Vietnamese."
]



EventQuery = Annotated[
        str,
        "An event-level query. The event-query must be in Vietnamese."
]