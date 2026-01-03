from typing import Annotated
from pydantic import Field


HANDLE_ID_ANNOTATION = Annotated[
    str,
    Field(description="This is the handle id from a tool call's result that you have previously invoked.")
]