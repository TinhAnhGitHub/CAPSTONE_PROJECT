from pydantic import BaseModel, Field
from typing import List


class WorkerBluePrint(BaseModel):
    name: str = Field(..., description="The specific name of the agent")
    description: str = Field(..., description="Detailed description of the agent's role, reasoning style, and tool usage")
    task: str = Field(..., description="The specific task that it must complete")
    tools: List[str] = Field(..., description="The available tools that the system offers")
    max_iterations: int = Field(3, description="Number of attempts before the agent gives up")

    def _get_string(self):
        return (
            f"WorkerBluePrint(\n"
            f"  name={self.name!r},\n"
            f"  description={self.description!r},\n"
            f"  task={self.task!r},\n"
            f"  tools={self.tools!r},\n"
            f"  max_iterations={self.max_iterations}\n"
            f")"
        )


class WorkersPlan(BaseModel):
    plan_detail: List[WorkerBluePrint] = Field(
        default_factory=list,
        description="Plan for these agents. Typically 1–3 agents. 2 is the sweet spot."
    )
    plan_summary: str = Field(..., description="The summary of the plan")
    reason: str = Field(..., description="The reasoning behind the plan")

    def _get_string(self):
        details_str = "\n".join(plan._get_string() for plan in self.plan_detail)
        return (
            f"WorkersPlan(\n"
            f"  plan_summary={self.plan_summary!r},\n"
            f"  reason={self.reason!r},\n"
            f"  plan_details=[\n{details_str}\n  ]\n"
            f")"
        )
