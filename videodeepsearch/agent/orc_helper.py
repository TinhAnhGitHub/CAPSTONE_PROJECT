from llama_index.core.agent import AgentWorkflow
from llama_index.core.workflow.handler import WorkflowHandler #type:ignore




def run_agent_orchestration(
    agent_orchestration: AgentWorkflow,
    user_message: str
) -> WorkflowHandler:
    handler = agent_orchestration.run(user_msg=user_message)
    return handler
    



