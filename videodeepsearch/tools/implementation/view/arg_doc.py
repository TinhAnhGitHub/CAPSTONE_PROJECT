"""
videodeepsearch/tools/implementation/view/arg_doc.py
"""
from typing import Literal, Annotated
from pydantic import Field

DetailLevel = Literal['quick', "detailed", "custom"]
slicing = str | int | list[int]


DETAIL_LEVEL_ANNOTATE = Annotated[
    DetailLevel,
    Field(
        description="Level of detail: 'quick' (counts only), 'detailed (top 20 with full infor)', 'custom' (use the custom slicing and filters). When using 'quick', 'detailed', other filters is ignored. When use 'custom', then the result will get filtered first, then get index/slicing"
    )
]

SLICING_ANNOTATION = Annotated[
    slicing,
    Field(
        description="Python slice string (e.g., '0:5', '::-1') or specific index (e.g. 1) or list of indices (e.g. [0, 2])."
    )
]

def parse_slicing(s: str | int | list[int]) -> slice | int | list[int]:
    if isinstance(s, list) or isinstance(s, int):
        return s
    if isinstance(s, str):
        if ":" in s:
            parts = [int(p) if p.strip() else None for p in s.split(":")]
            return slice(*parts)
        try:
            return int(s)
        except ValueError:
            raise ValueError(f"Invalid slicing format: {s}")
    return s