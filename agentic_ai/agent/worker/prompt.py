from typing import  Annotated
from llama_index.core.prompts import PromptTemplate, RichPromptTemplate



MAKE_DECISION_PROMPT: Annotated[
    PromptTemplate,
    "This prompt template will help the agent choosing to use the function directly, or to spawn a code which can execute more complex task"
] = PromptTemplate(
    """
    You must decide the best approach to solve the user's request.
    
    **OPTION 1: tools** - Use when:
    - Single operation (search, retrieve, convert)
    - Direct function call works (e.g., "find images of cars", "get video metadata")
    - No intermediate processing needed
    - Simple data retrieval or lookup

    **OPTION 2: code** - Use when:
    - Multiple sequential operations needed
    - Requires loops, conditions, or aggregation
    - Need to process/filter/transform results
    - Combining outputs from multiple tools
    - Complex logic like "find X, then for each result check Y, filter by Z"

    **Available tools:**
    {tool_descriptions}
    """
)


FEW_SHOTS_PROMPT: list[tuple[str, str]] = [
    (
        """
        This example will demonstrate the usage of a simple visual query. The user demand will be: Hãy tìm cho tôi khoảnh khắc của một con chó đang chơi cùng các bạn nhỏ trong 1 khu vườn
        """,
        """
        <execute>

        async def collect_dog_moments(user_str: str):
            enhanced_visual = await enhance_visual_query(raw_query=user_str, variants=[])
            
            results: list[list[ImageObjectInterface]] = []
            top_k: int = 10 # 10 images return each query
            for query in enhanced_visual:
                images_retrieved = await get_images_from_visual_query(visual_query=query, top_k=top_k)
                results.append(images_retrieved)
            return results
        
        results = await collect_dog_moments("Find me a moment of a dog playing with his friends in a garden.")
        print(f"{results=}")
        results
        <execute>
        """
    ),
    (
        """
        This example highlights a caption-based retrieval flow. The user demand will be: Hãy tìm các khoảnh khắc tại chợ nổi với nhiều ghe chở trái cây và người mua bán tấp nập
        """,
        """
        <execute>

        async def collect_floating_market_views(user_str: str):
            enhanced_caption_queries = await enhance_textual_query(raw_query=user_str, variants=["góc rộng", "cận cảnh"])

            all_results: list[list[ImageObjectInterface]] = []
            for query in enhanced_caption_queries:
                images = await get_images_from_caption_query(
                    caption_query=query,
                    top_k_each_request=8,
                    top_k_final=8,
                    weights=None
                )
                all_results.append(images)
            return all_results

        caption_results = await collect_floating_market_views("Tìm khoảnh khắc tại chợ nổi với nhiều ghe chở trái cây và người mua bán tấp nập.")
        print(f"{caption_results=}")
        caption_results
        <execute>
        """
    ),
    (
        """
        This example illustrates event-level segment discovery. The user demand will be: Hãy tìm cho tôi những phân đoạn có cảnh rượt đuổi xe hơi trong đêm mưa
        """,
        """
        <execute>

        async def collect_rainy_chase_segments(user_prompt: str):
            refined_queries = await enhance_textual_query(raw_query=user_prompt, variants=["bối cảnh", "góc quay"])

            segment_batches: list[list[SegmentObjectInterface]] = []
            for query in refined_queries:
                segments = await get_segments_from_event_query(
                    event_query=query,
                    top_k_each_request=6,
                    top_k_final=6,
                    weights=(0.8, 0.2)
                )
                segment_batches.append(segments)
            return segment_batches

        chase_segments = await collect_rainy_chase_segments("Tìm những cảnh rượt đuổi xe hơi trong đêm mưa.")
        print(f"{chase_segments=}")
        chase_segments
        <execute>
        """
    ),
    (
        """
        This example demonstrates multimodal image retrieval. The user demand will be: Hãy tìm những bức ảnh về lễ hội đèn lồng với nhiều ánh sáng ấm áp
        """,
        """
        <execute>

        async def gather_lantern_festival_images(visual_en: str, caption_vi: str):
            visual_queries = await enhance_visual_query(raw_query=visual_en, variants=["wide angle", "crowd view"])
            caption_queries = await enhance_textual_query(raw_query=caption_vi, variants=["không khí", "màu sắc"])

            combined_results: list[list[ImageObjectInterface]] = []
            for visual_query, caption_query in zip(visual_queries, caption_queries):
                multimodal = await get_images_from_multimodal_query(
                    visual_query=visual_query,
                    caption_query=caption_query,
                    top_k_each_request=5,
                    top_k_final=5,
                    weights=(0.6, 0.3, 0.1)
                )
                combined_results.append(multimodal)
            return combined_results

        lantern_results = await gather_lantern_festival_images(
            "Lantern festival at night with warm orange lights and joyful crowds.",
            "Hãy tìm khoảnh khắc lễ hội đèn lồng với ánh sáng ấm áp và người dân mỉm cười."
        )
        print(f"{lantern_results=}")
        lantern_results
        <execute>
        """
    ),
    (
        """
        This example chains visual retrieval with similarity expansion. The user demand will be: Hãy tìm thêm những khoảnh khắc tương tự một người đang lướt sóng lúc bình minh
        """,
        """
        <execute>

        async def expand_sunrise_surfing_moments(user_prompt: str):
            visual_queries = await enhance_visual_query(raw_query=user_prompt, variants=["aerial view", "close up"])

            similarity_pools: list[list[ImageObjectInterface]] = []
            for query in visual_queries:
                seeds = await get_images_from_visual_query(visual_query=query, top_k=6)
                if not seeds:
                    similarity_pools.append([])
                    continue
                anchor = seeds[0]
                related = await find_similar_images_from_image(reference_image=anchor, top_k=12)
                similarity_pools.append(related)
            return similarity_pools

        surfing_neighbors = await expand_sunrise_surfing_moments("Surfer catching a wave at sunrise with golden light.")
        print(f"{surfing_neighbors=}")
        surfing_neighbors
        <execute>
        """
    ),
]




CODE_ACT_PROMPT = RichPromptTemplate("""
You are a a professional in Video Deep Search. Given a list of tools description, and the user's demand, you are able to write an executable complex Python code to satisfy the given problems.

Rules:
- Python code wrapped in <execute>...</execute> tags that provides the solution to the task, or a step towards the solution. Any output you want to extract from the code should be printed to the console.
- If the previous code execution can be used to respond to the user, then respond directly (typically you want to avoid mentioning anything related to the code execution in your response).

## Response Format:

                                     
Example of proper code format:
{% for instruction, example in few_shot_prompts%}
Example scenario: {{ instruction }}
Example code structure: 
{{example}}
{% endfor %}

In addition to the Python Standard Library and any functions you have already written, you can use the following functions:
{tool_descriptions}

Variables defined at the top level of previous code snippets can be also be referenced in your code.

## Final Answer Guidelines:
- When providing a final answer, focus on directly answering the user's question
- Avoid referencing the code you generated unless specifically asked
- Present the results clearly and concisely as if you computed them directly
- If relevant, you can briefly mention general methods used, but don't include code snippets in the final answer
- Structure your response like you're directly answering the user's query, not explaining how you solved it

Reminder: Always place your Python code between <execute>...</execute> tags when you want to run code. You can include explanations and other content outside these tags
""")

code_act_prompt = CODE_ACT_PROMPT.partial_format(few_shot_prompts=FEW_SHOTS_PROMPT)

if __name__ == "__main__":
    print(code_act_prompt)