"""
This will represent the state of the small worker agents, where it will persist its results, and pass it to the Global Shared State for other agents to see.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any
from llama_index.core.agent.workflow import ToolCallResult

from videodeepsearch.tools.schema.artifact import BaseArtifact
from videodeepsearch.agent.worker.planner.schema import WorkersPlan


class Evidence(BaseModel):
    description: str = Field(..., description="Summarize the evidence found")
    references: list[str] = Field(default_factory=list, description="The information derived from the artifact")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: None | str = Field(..., description="A piece of information that you want to address/summarize, or communicate to the orchestrator agents")

class WorkerFinding(BaseModel):
    summary: str = Field(..., description="Detailed summary of the findings, including the results, and the process of findings. Basically story telling")
    confidence_score: int = Field(..., le=5, ge=1, description="The confidence score based on 5-rank star")
    found_answer: bool 
    evidences: list[Evidence] = Field(default_factory=list)



SUB_ORCHESTRATOR_STATE_KEY = 'SUB_ORCHESTRATOR_STATE'
class SubOrchestrationState(BaseModel):
    """
    Orchestrator - Workflow
    Single source of truth for all agents
    """
    user_demand: str = Field(...)
    started_at: str = Field(default_factory=datetime.now().isoformat)
    completed_at: str| None
    
    workers: list[str] = Field(default_factory=list, description="All worker names")
    worker_plans: WorkersPlan | None = Field(None)
    findings: dict[str, list[WorkerFinding]] = Field(default_factory=dict, description="The finding of the worker agents while working")
    tool_logs: dict[str, list[ToolCallResult]] = Field(default_factory=dict)
    all_evidence: list[Evidence] = Field(default_factory=list, description="The list of evidence used in the orchestration")

    answer_found: bool | None
    confidence: int | None = Field(None, le=5, ge=1)
    final_answer: str | None = Field(None)


    def add_worker_finding(
        self,
        worker_name: str,
        summary: str,
        confidence: int,
        evidences: list[Evidence],
        found_answer: bool = False
    ):
        finding = WorkerFinding(
            summary=summary,
            confidence_score=confidence,
            found_answer=found_answer,
            evidences=evidences   
        )
        self.findings.setdefault(worker_name,[])
        self.findings[worker_name].append(finding)

        for ev in evidences:
            self.all_evidence.append(ev)

    def add_tool_results(
        self,
        worker_name: str,
        tool_results: list[ToolCallResult]
    ):
        self.tool_logs.setdefault(worker_name, [])
        self.tool_logs[worker_name].extend(tool_results)
    

def create_sub_orchestrator_initial_state(
    user_query: str,
) -> dict:
    
    return SubOrchestrationState(
        user_demand=user_query,
        worker_plans=None,
        workers=[],
        completed_at=None,
        findings={},
        all_evidence=[],
        answer_found=None,
        confidence=None,
        final_answer=None,
    ).model_dump(mode='json')



    