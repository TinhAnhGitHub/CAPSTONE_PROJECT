"""
videodeepsearch/tools/implementation/util/arg_alias.py
"""
from typing import Annotated
from typing import Annotated

WindowSeconds = Annotated[
    float,
    "Time window around artifact (± seconds for transcript snippet)."
    "IF 10 seconds, then we will span window 5 forward, 5 backward"
]


