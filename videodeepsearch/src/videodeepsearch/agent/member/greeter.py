"""
Greeter agent
"""
from agno.agent import Agent  
from agno.models.openrouter import OpenRouter
from agno.db.async_postgres import AsyncPostgresDb

DESCRIPTION: str = """

"""

INSTRUCTIONS: list[str] = [
    
]

def get_greeting_agent(
    session_id: str,
    user_id: str,
    model: OpenRouter,
    db: AsyncPostgresDb,
    
) -> Agent:
    
    greeting_agent = Agent(
        name='Greeter_Agent',
        model=model,
        user_id=user_id,
        session_id=session_id,
        db=db,
        
        add_session_state_to_context=True,
        enable_agentic_state=True,
        update_memory_on_run=True,
        add_memories_to_context=True,
        
        enable_session_summaries=True,
        add_session_summary_to_context=True,
        search_past_sessions=True,
        
        markdown=True,
        add_datetime_to_context=True,
        description=DESCRIPTION,
        instructions=INSTRUCTIONS
    )
    return greeting_agent
