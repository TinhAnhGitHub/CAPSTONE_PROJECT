"""
This class will represent the interface of the results. Only a subset of information from the result are exposed to the agent to avoid token overloading. 

Any tool call that return the artifact, the agent only see some information. The actual result is persist in the Agent context, and there will be a subset of tools that allow the agent to retrieve them "intelligently" (base on score, based on range, e.tc..).
"""
# from typing import Type
# from uuid import uuid4
# from pydantic import BaseModel, Field

# from .artifact import BaseArtifact

# class SearchResultInterface(BaseModel):
#     handle_id: str = Field(default_factory=lambda: f"hdl_{uuid4().hex[:12]}")
#     total_count: int = Field(..., description="The total amount of return item")
#     type_artifact: Type[BaseArtifact] = Field(..., description="The kind of artifact return")
#     preview_items: list[BaseArtifact] = Field(..., description="Top 5 returned results")
    
#     page_size: int = Field(default=)

