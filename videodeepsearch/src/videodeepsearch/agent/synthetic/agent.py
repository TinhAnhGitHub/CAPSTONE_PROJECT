import random
from typing import Literal

from agno.agent import Agent
from agno.models.base import Model
from agno.tools import Toolkit

from videodeepsearch.agent.synthetic.prompt import (
    QA_GENERATOR_SYSTEM_PROMPT,
    QA_GENERATOR_INSTRUCTIONS,
)
from videodeepsearch.toolkit.video_metadata import VideoMetadataToolkit
from videodeepsearch.toolkit.utility import UtilityToolkit
from videodeepsearch.toolkit.search import VideoSearchToolkit


def get_qa_generator_agent(
    agent_name: str,
    user_id: str,
    model: Model,
    video_metadata_toolkit: VideoMetadataToolkit,
    utility_toolkit: UtilityToolkit,
    search_toolkit: VideoSearchToolkit | None = None,
    num_questions: int = 5,
    question_types: list[
        Literal["visual", "temporal", "audio", "multi_hop", "factoid", "descriptive"]
    ]
    | None = None,
    tool_call_limit: int = 30,
) -> Agent:
    """Create a standalone Q&A generation agent for synthetic data.

    This agent randomly explores videos, gathers context using video utility
    and metadata tools, and generates question-answer pairs with ground truth.

    Args:
        agent_name: Unique identifier for this agent instance (e.g., "qa_gen_01").
        user_id: User ID to filter videos for generation.
        model: The LLM model to use for generation.
        video_metadata_toolkit: Toolkit for listing videos, metadata, timelines.
        utility_toolkit: Toolkit for ASR context, adjacent segment navigation.
        search_toolkit: Optional toolkit for content-based video search.
        num_questions: Target number of Q&A pairs to generate (default 5).
        question_types: List of question types to generate. If None, generates all types.
        tool_call_limit: Maximum number of tool calls per session (default 30).

    Returns:
        Agent instance configured for Q&A generation.

    Example:
        ```python
        from videodeepsearch.agent.synthetic import get_qa_generator_agent
        from videodeepsearch.toolkit.factories import (
            make_video_metadata_factory,
            make_utility_factory,
        )

        agent = get_qa_generator_agent(
            agent_name="qa_gen_session_01",
            user_id="user_123",
            model=gpt_4o,
            video_metadata_toolkit=make_video_metadata_factory(postgres, minio)(),
            utility_toolkit=make_utility_factory(postgres, minio)(),
            num_questions=10,
        )

        # Run the agent
        result = await agent.arun("Generate Q&A pairs from available videos")
        ```
    """
    toolkits: list[Toolkit] = [video_metadata_toolkit, utility_toolkit]
    if search_toolkit:
        toolkits.append(search_toolkit)

    if question_types:
        types_instruction = f"\n\n## Target Question Types\nFocus on generating these types: {', '.join(question_types)}"
    else:
        types_instruction = "\n\n## Target Question Types\nGenerate a balanced mix of: visual, temporal, audio, multi_hop, factoid, and descriptive questions."

    enhanced_instructions = QA_GENERATOR_INSTRUCTIONS + [
        f"Generate approximately {num_questions} question-answer pairs in this session.",
        types_instruction,
    ]

    system_prompt = QA_GENERATOR_SYSTEM_PROMPT.format(
        agent_name=agent_name,
        video_ids='\n'.join(
            [
                "02d242459a690605ee3a8ddf",
                "0e64f1c0da591ca67f07b7f9",
                "0f48acd4ac783dfbdee85468",
                "1e1d300356360ed84020821c",
                "3636d10a2ad4787733c9700d",
                "4a081d6f16c83d089f67161b",
                "533914541945c2060c128da3",
                "92ba4b2e27f460945fded9e5",
                "946330031ead69b21354d038",
                "9b17f473300a5436f0a053be",
                "b1abb34af1bc67cb712d5ffb",
                "c510fac771767405c891bf64",
                "c98019fd17ff4420ea47eee7",
                "eee3534844edab3ebb4d6ceb",
                "f784df5c6c00eb123f9820dc",
            ]
        ),
    )

    return Agent(
        name=agent_name,
        role="Generate synthetic Q&A pairs from video content with ground truth annotations",
        model=model,
        system_message=system_prompt,
        instructions=enhanced_instructions,
        tools=toolkits,
        tool_call_limit=tool_call_limit,
        add_session_state_to_context=False,
        enable_agentic_state=False,
        add_history_to_context=False,
        update_memory_on_run=False,
        enable_session_summaries=False,
        markdown=False,  # JSON output preferred
        # Reliability
        retries=1,
        delay_between_retries=1,
        # Streaming
        debug_mode=True,
        debug_level=1,
        stream_events=True,
        stream=True,
    )

