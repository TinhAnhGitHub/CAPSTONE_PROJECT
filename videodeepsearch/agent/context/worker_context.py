# context/worker_context
from pydantic import BaseModel, Field
from typing import Union
from llama_index.core.llms import ChatMessage
from llama_index.core.agent.workflow import ToolCall, ToolCallResult
from datetime import datetime
from typing import Sequence, cast
from uuid import uuid4

from videodeepsearch.tools.base.middleware import ResultStore, DataHandle
from videodeepsearch.tools.base.schema import BaseInterface, ImageInterface, SegmentInterface


class EvidenceItem(BaseModel):
    evidence_id: str = Field(default_factory=lambda: str(uuid4()))
    source_worker_name: str = Field(..., description="Which worker produced this")
    source_tool_call: ToolCall = Field(..., description="Which tool call generated it")
    artifacts: Sequence[Union[ImageInterface, SegmentInterface]] = Field(...)
    confidence_score: int = Field(..., ge=1, le=10)
    related_video_ids: list[str] = Field(default_factory=list)
    claims: str 

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        """
        Represent the evidence
        """
        line_artifacts = []
        for artifact in self.artifacts:
            line_artifacts.append(
                artifact.detailed_representation()
            )
        line_artifacts_str = '\n'.join(line_artifacts)
        
        template = f"""
        From tool:{self.source_tool_call}
        Artifacts: 
        {line_artifacts_str}
        Confidence score: {self.confidence_score}
        Claims: {self.claims}
        """

        return template


class SmallWorkerContext(BaseModel):
    """
    Context for the worker agent
    Philosophy: Workers are stateless executors - receive task, use tools, persist results, and report back to the orchestrator
    """

    worker_id: str = Field(
        default_factory=lambda: str(uuid4())
    )
    worker_agent_name: str = Field(...)
    
    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="The chat history"
    )
    
    task_objective: str = Field(
        ...,
        description="Clear and concrete task for this worker"
    )

    raw_result_store: ResultStore = Field( 
        default_factory=lambda: ResultStore(),
        description=(
            "Chronological log of tool invocations, where each entry maps a ToolCall "
            "to its corresponding parsed BaseModel output."
        )
    )

    evidences: list[EvidenceItem] = Field( 
        default_factory=list,
        description="Evidence items this worker deems relevant for resolving the task"   
    )

    class Config:
        arbitrary_types_allowed = True
    
        