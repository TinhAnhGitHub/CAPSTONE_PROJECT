from pydantic import BaseModel, Field
from typing import Literal
class ToolsOrCodeDecision(BaseModel):
    reason: str = Field(..., description="The reason why you obtain this decision")
    decision: Literal['tools', 'code']