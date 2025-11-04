# Worker Agent Prompt Stress Scenarios

These scenarios outline medium-complexity tasks that encourage the Worker Code Video Agent
to prefer the code-act pathway and orchestrate multiple tools. Each item highlights the
expected plan, tool usage, and why the decision prompt should favour `code`.

## 1. Multistage Festival Montage
- **User ask**: "Build a highlight reel of night-time lantern festivals showing wide shots, then close-ups." 
- **Why `code`**: Requires combining results from visual and caption search, deduplicating by video id, and ordering clips.
- **Plan sketch**:
  1. Enhance both visual and textual queries for coverage.
  2. Fetch candidate images/segments from `get_images_from_multimodal_query` and `get_segments_from_event_query`.
  3. Rank and interleave clips by lighting intensity metadata.
  4. Return a structured montage brief.
- **Extra tool prompts**: Encourage `find_similar_images_from_image` for fallback enrichment.

## 2. Surf Competition Comparator
- **User ask**: "Compare sunrise surfing shots with sunset filming angles and summarise key differences."
- **Why `code`**: Needs two parallel searches, summarisation, and contrastive analysis.
- **Plan sketch**:
  1. Generate variant queries ("sunrise aerial", "sunset backlit").
  2. Loop through results from `get_images_from_visual_query`; extract top frames.
  3. Use `find_similar_images_from_image` to expand each bucket.
  4. Aggregate descriptors (colour palette, camera angle, crowd density) and summarise differences.

## 3. Crowd Safety Audit
- **User ask**: "Identify stadium scenes with overcrowding, then cross-check for emergency exits visibility."
- **Why `code`**: Requires filtering by thresholds and conditional branching.
- **Plan sketch**:
  1. Call `enhance_visual_query` with safety-focused variants.
  2. Fetch segments via `get_segments_from_event_query` with density metadata.
  3. For each segment, route to `get_images_from_caption_query` targeting "exit" keywords.
  4. Flag segments lacking exits and compile alert list.

## 4. Weather Anomaly Reporter
- **User ask**: "List videos where sunny forecasts led to rain on-site, including timestamps."
- **Why `code`**: Cross-tool correlation between forecast metadata and actual visual evidence.
- **Plan sketch**:
  1. Query `get_segments_from_event_query` for weather transitions.
  2. Pair results with `search_external_weather_metadata` (tool stub) to fetch forecast notes.
  3. Filter mismatches, store as structured JSON, respond with summary.

## 5. Fashion Trend Timeline
- **User ask**: "Trace the evolution of streetwear neon outfits across 2020-2024."
- **Why `code`**: Time-series aggregation and iterative expansion.
- **Plan sketch**:
  1. Iterate over yearly buckets, calling `get_images_from_caption_query` with year filters.
  2. For representative frames, use `find_similar_images_from_image` to cluster styles.
  3. Compute counts per year, detect emerging colours, output timeline narrative.

## Prompt Tweaks Reference
- **Decision prompt**: Highlight criteria for medium/complex orchestration and require justification referencing tool combos.
- **Code-act prompt**: Ask the agent to emit a plan header and encourage tool fan-out/fan-in patterns plus validation steps.
- **System prompt**: Reinforce reuse of intermediate results, fallback handling, and polished final messaging.

Use these scenarios when evaluating prompt effectiveness or demonstrating the agent's high-leverage behaviours without writing full executable tests.
