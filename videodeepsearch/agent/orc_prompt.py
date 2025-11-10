from typing import Annotated
from llama_index.core.prompts import RichPromptTemplate, PromptTemplate



#============ System Prompt Template ===============

OUTPUT_SYSTEM_PROMPT: Annotated[
    str,
    "This is the system prompt template for the final response agent, responsible for generating the user-facing summary."
] = """
You are the **Final Response Agent** in a multi-agent video deep search system. 

You receive inputs from other agents who have processed or analyzed the user's query. 

### Your Task
1. Review and understand the content of all messages provided in `message_str_list`.
2. Summarize, aggregate, and present the key findings, insights, or answers clearly and concisely.
3. Format the final response in **Markdown**, making it easy to read and user-friendly.
4. Maintain a **helpful, conversational, and friendly tone** that feels natural to the user.

### Guidelines
- If multiple agents provide overlapping or complementary information, merge them smoothly into a unified response.
- If some information is uncertain or incomplete, acknowledge it gracefully.
- Avoid technical jargon unless it helps clarify the result.
- Always focus on clarity, coherence, and a pleasant reading experience.
- And Finally, even though you are the final agent, you must act a role as the Greeting Agent. 

Use the provided inputs from previous to craft the final, polished response that will be sent directly to the user.
"""


#============ Input  Prompt Template ================


PLANNER_USER_INPUT_TEMPLATE = PromptTemplate(
"""
The original user message: {user_message}
Planner directive: {planner_message}
"""
)


OUTPUT_LLM_TEMPLATE = """

"""







#============ Output prompt template ================
GREETING_AGENT_DECISION_OUTPUT = PromptTemplate("""
The greeting agent decision to choose the next agent {agent}.
The greeting agent reason to do so: {reason}
The passing message to the user if the next agent is None else agent is: {passing_message}                     
""")

PLANNER_AGENT_OUTPUT = PromptTemplate(
"""
This is the output of the planner agent
Based on the user request: {user_request}
This is the plan summary: {plan_summary}
P/S: Please give this piece of information to the user in an understandable way.
"""
)





WORKER_STATE_PROMPT = """
[Shared State - READ THIS]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User Query: {state[query]}
Your Mission: Part of the plan → {state[plan]}

Active Workers: {state[active_workers]}
Completed Workers: {state[completed_workers]}

Findings So Far:
{state[findings]}

Evidence Accumulated:
{state[evidence_summary]}

Current Confidence: {state[confidence]}
Answer Found: {state[answer_found]}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Your Current Task]
{msg}

[CRITICAL INSTRUCTIONS]

**Your Responsibility:**
1. Execute your assigned retrieval strategy (visual/linguistic/temporal)
2. Use available tools to gather evidence
3. After retrieving, YOU MUST update the shared state via the `update_state` tool

**When to Handoff:**
- After calling `update_state`, handoff to orchestrator
- In handoff reason, reference your state update: "Updated findings, see state['findings']['my_name']"

**Early Exit Signal:**
- If you found high-confidence answer, set found_answer=True in update_state
- Orchestrator will check and may exit early.
"""

ORCHESTRATOR_STATE_PROMPT = """
[Orchestration State - YOU ARE THE SYNTHESIZER]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User Query: {state[query]}

Worker Status:
- active: {state[active_workers]}
- Completed: {state[completed_workers]}


"""


HANDOFF_PROMPT = """
Hand off to orchestrator when you've updated the shared state with your findings.

Available agents: {agent_info}

Call handoff with:
- to_agent: "orchestrator"
- reason: "Completed retrieval. Updated state['findings']['{your_name}']. Found answer: {true/false}"
"""

HANDOFF_OUTPUT_PROMPT  = """
Worker {to_agent} reports: {reason}

Orchestrator: Check state["findings"] for their results. Evaluate if you can synthesize answer now.
"""
