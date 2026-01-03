from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal
from datetime import datetime
from uuid import uuid4

from llama_index.core.agent.workflow import ToolCall
from llama_index.core.llms import ChatMessage

from videodeepsearch.tools.base.middleware import ResultStore

from .worker_context import EvidenceItem


class WorkerResult(BaseModel):
    worker_name: str = Field(...)
    task_objective: str = Field(...)
    worker_chat_history: list[ChatMessage] = Field(default_factory=list)
    
    raw_result_store: ResultStore = Field(
        default_factory=lambda: ResultStore(),
        description=(
            "Chronological log of tool invocations, where each entry maps a ToolCall "
            "to its corresponding parsed BaseModel output."
        )
    )
    
    evidences: list[EvidenceItem] = Field(
        default_factory=list,
        description="The related evidence that the worker thinks that it resolve the task"   
    )

    result_summary: str = Field(
        ...,
        description=(
            "Comprehensive summary of the worker's full task execution process:"
            "the objective, reasoning steps, tools invoked, insights gained from tool outputs, and how these led to the final result"
        )
    )

    completed_at: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True

class VideoContext(BaseModel):
    video_id: str = Field(...)
    findings: list[str] = Field(default_factory=list)

    def generate_context(self) -> str:
        if not self.findings:
            return f"Video ID: {self.video_id}\nNo findings."

        lines = [f"Video ID: {self.video_id}", "Findings:"]
        lines += [f"- {item}" for item in self.findings]
        return "\n".join(lines)

class OrchestratorContext(BaseModel):
    """
    Persistent Context for sub-orchestrator agent
    """

    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Persistent session ID"
    )

    user_demands: list[str] = Field(
        default_factory=list,
        description="The user's demand overtime"
    )

    video_context: dict[str, VideoContext] = Field(
        default_factory=dict,
        description="This is the accumulated video context, persist the user video asking questions."
    )

    history_worker_results: list[list[WorkerResult]] = Field(
        default_factory=list,
        description="The history of all the worker task"
    )

    # summarize_works: list[str] = Field(default_factory=list)

    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        description="The orchestrator chat history"
    )


    class Config:
        arbitrary_types_allowed = True
    

    def update_video_context(self, video_id: str, new_finding: str):
        video_context = self.video_context.get(video_id)
        if video_context is None:
            video_context = VideoContext(video_id=video_id, findings=[])
        video_context.findings.append(new_finding)
        self.video_context[video_id] = video_context

    def add_to_latest_worker_results(
        self,
        worker_result: WorkerResult
    ):
        self.history_worker_results[-1].append(worker_result)

    
    def get_agent_evidence(self, agent_worker_name: str) -> str:
        latest_worker_results = self.history_worker_results[-1]
        result = list(
            filter(lambda x: x.worker_name == agent_worker_name, latest_worker_results)
        )
        if not result:
            return f"The agent name: {agent_worker_name} is not exist in the evidences history."
        results = [
            r.result_summary for r in result
        ]
        separator = "\n" + "=" * 50 + "\n"
        return separator.join(results)
    

    
    def get_all_worker_agent_name(self) -> str:
        agent_name_list = set()
        for session in self.history_worker_results:
            for worker_result in session:
                agent_name_list.add(worker_result.worker_name)
        
        return  f"Here are all the agent names:\n {', '.join(list(agent_name_list))}"
    

def prepare_init_ctx_each_run(
    current_orc_ctx: OrchestratorContext | None,
    session_id: str,
    user_demand: str,
) -> OrchestratorContext:
    """
    Prepare the Orc context
    if None -> create new ctx, and prepare some of the field
    If exist, then create like a new history
    """

    if current_orc_ctx is None:
        orc_context = OrchestratorContext(
            session_id=session_id,
            user_demands=[user_demand],
            history_worker_results=[[]]
        )
        return orc_context
    
    current_orc_ctx.user_demands.append(user_demand)
    current_orc_ctx.history_worker_results.append([])
    return current_orc_ctx

