from tqdm import tqdm
import json
import requests
from mlflow.genai.judges import make_judge
from typing import Literal
import mlflow
import re
from typing import cast

from mlflow.entities import  Trace, SpanType


MLFLOW_TRACKING_URI = "http://100.113.186.28:5000"
EXPERIMENT_NAME = "vds-agent-validation"
EXPERIMENT_ID = "6"
TRACE_FILTER = 'trace.text LIKE "%I want%"'
LLM_URL = "http://localhost:8080/v1/chat/completions"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def extract_json(content: str):
    if content.startswith("```json"):
        content = content.removeprefix("```json").strip()

    if content.startswith("```"):
        content = content.removeprefix("```").strip()

    if content.endswith("```"):
        content = content.removesuffix("```").strip()
      
    return content



SEGMENT_EXTRACT_PROMPT = """
Extract every video ID and its referenced time segments from the agent’s response, and return them as a single JSON object.

Requirements:

* Use the video ID as the JSON key.
* The value must be a list of unique time segments for that video.
* Each segment must be formatted as:
  `["HH:MM:SS", "HH:MM:SS"]`
* Convert all second-based values into human-readable timestamps.
* Remove duplicate segments.
* If a video appears multiple times, merge all segments under the same video ID.
* Preserve the original order of appearance.
* Do not include any explanation, commentary, or extra text.
* Return only valid JSON.
* If no video segments are found, return empty dict.

Expected format:

```json
{
  "<video_id>": [
    ["00:00:00", "00:01:12"],
    ["00:05:23", "00:06:12"]
  ],
  "<another_video_id>": [
    ["00:01:42", "00:03:05"]
  ]
}
```
"""


def llm_invoke(agent_response: str, trace_id: str) -> dict:
    print(f"Running on trace id: {trace_id=}")

    escaped_response = agent_response.replace("{", "{{").replace("}", "}}")
    
    prompt_llm = f"""
    Here is the agent's response content:
    ######
    {escaped_response}
    ######
    """ + SEGMENT_EXTRACT_PROMPT
    
    retries = 3
    
    json_dict = None
    
    for attempt in range(retries):
        try:
            response = requests.post(
                LLM_URL, 
                json={
                    "messages": [{"role": "user", "content": prompt_llm}], "max_tokens": 32000,
                },
                timeout=750
            )
            
            response.raise_for_status()
            
            raw_content = response.json().get("choices", [])[0].get("message", {}).get("content", "")
            clean_raw_content = extract_json(content=raw_content)
            json_dict = json.loads(clean_raw_content)
        except Exception as e:
            print(response.json())
            print(f"Attempt {attempt + 1}: Error: {e}")
      
    if json_dict is None:
      print(f"Max retries for {trace_id=}. Return empty dict")
      return {}
    
    print(f"Successfull extraction for {trace_id=}")
    
    return json_dict

def extract_agent_response_from_trace(trace: Trace) -> str:
    agent_response = trace.data.response
    return cast(str, agent_response)

 
def main():
    traces = cast(
        list[Trace],
        mlflow.search_traces(
            locations=[EXPERIMENT_ID],
            filter_string=TRACE_FILTER,
            return_type='list'
        )
    )

    output_file = "extracted_segments.json"
    existing_results: dict[str, dict] = {}
    try:
        with open(output_file, "r") as f:
            existing_results = json.load(f)
        print(f"Loaded {len(existing_results)} existing results")
    except FileNotFoundError:
        print("No existing results found, starting fresh")

    print(f"Found {len(traces)} traces")

    extracted_segments: dict[str, dict] = existing_results.copy()

    for trace in tqdm(traces, desc="Processing trace..."):
        trace_id = trace.info.trace_id

        if trace_id in existing_results and existing_results[trace_id]: #skip already 
            print(f"Skipping already processed trace: {trace_id}")
            continue

        agent_response = extract_agent_response_from_trace(trace)

        segments = llm_invoke(agent_response, trace_id)

        extracted_segments[trace_id] = segments

    output_file = "extracted_segments.json"
    with open(output_file, "w") as f:
        json.dump(extracted_segments, f, indent=2)

    print(f"Saved {len(extracted_segments)} extracted segments to {output_file}")

    return extracted_segments


if __name__ == "__main__":
    main()
        
        