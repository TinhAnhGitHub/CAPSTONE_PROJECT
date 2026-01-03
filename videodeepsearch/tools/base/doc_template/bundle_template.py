from ..types import BundleRoles, BundleRole, BundleSpec

QUERY_ANALYZER_SPEC = BundleRole(
    name="Query Analyzer",
    description=(
        "Tools for enhancing vague or underspecified user queries before embedding or retrieval. "
        "Specializes in expanding visual iantic intent (ntent (images/video) and textual/semcaptions, events, scenes) "
        "by generating richer, contrastive, or context-aware query variants using LLMs. "
        "Used to improve retrieval relevance when raw queries lack sufficient cues for embedding models."
    ),
    purpose="Turn ambiguous or short user queries into multiple high-quality, retrieval-ready variants"
)

SEMANTIC_SEARCHER_SPEC = BundleRole(
    name="Semantic searcher",
    description=(
        "High-precision retrieval tools that execute semantic, visual, caption-based, event-based, and multimodal searches"
        "against Milvus vector indexes (images + video segments). Supports dense, sparse, hybrid, and multimodal fusion"
        "with weighted reranking, plus image-by-image similarity search."
    ),
    purpose="Given an enhanced query (text or image), return the top-k most relevant images or video segments "
            "from the user's indexed library using state-of-the-art embedding similarity and hybrid retrieval.",
    typical_inputs_from=["Query Analyzer"],   
)

TRANSCRIPT_ANALYZER_SPEC = BundleRole(
    name="Transcript Analyzer",
    description=(
        "Tools that extract and format relevant spoken context (ASR transcript) "
        "around a given video moment  either a retrieved segment or a specific image/frame. "
        "Provides time-aligned, human-readable transcript snippets with configurable "
        "±window_seconds for grounding visual or semantic search results in speech."
    ),
    purpose="Fetch and format ASR transcript context (±N seconds) around any retrieved segment or timestamped image.",
    typical_inputs_from=["Semantic Searcher", "Video Navigator"],
)

WORKER_RESULT_INSPECTOR_SPEC = BundleRole(
    name="Result inspector tools",
    description=(
        "Tools that let worker agents introspect, filter, slice, and summarize previously stored "
        "search/retrieval results based on the DataHandle or orchestrator state. Enables viewing ranked evidence, "
        "extracting video-specific context, fetching raw persisted data by handle, "
        "and generating quick statistics (by video, score, or time)."
    ),
    purpose="Give worker agents safe, read-only access to past tool outputs and session context, so they can reason about results, verify relevance, or build summaries without re-running expensive searches.",
    typical_inputs_from=["Semantic Searcher"],
)

ORCHESTRATOR_WORK_INSPECTOR_SPEC = BundleRole(
    name="Orchestrator Inspector",
    description=(
        "Read-only tools exclusive to the orchestrator agent. "
        "Allows inspection of per-worker evidence accumulation and access to historical synthesized summaries "
        "across the current reasoning session."
    ),
    purpose="Enable the orchestrator to monitor progress, retrieve any worker's collected evidence, "
            "and review past round summaries without asking workers directly.",
)

WORKER_EVIDENCE_MANAGER_SPEC = BundleRole(
    name="Worker Evidence Manager",
    description=(
        "Tools exclusive to worker agents for persisting high-confidence results as scored evidence "
        "and submitting their final task summary when work is complete."
    ),
    purpose="Allow workers to selectively save evidence from tool results + mark task completion "
            "by pushing all accumulated evidence + summary to the shared orchestrator context.",
    typical_inputs_from=["Worker Result Inspector", "Transcript Analyzer", "Video Navigator"],
)

ORCHESTRATOR_EVIDENCE_MANAGER_SPEC = BundleRole(
    name="Orchestrator Evidence Manager",
    description=(
        "Orchestrator-only tools for enriching per-video findings and writing the final session synthesis "
        "or round summaries based on all collected worker evidence."
    ),
    purpose="Enable orchestrator to progressively update video-level insights and produce "
            "the ultimate user-facing report or intermediate round summaries."
)

VIDEO_NAVIGATOR = BundleRole(
    name="Video Navigator",
    description=(
        "Tools for agents to browse, navigate, and extract content from indexed videos. "
        "Includes: resolving parent video from any segment/image, fetching full ASR transcripts, "
        "retrieving all or adjacent segments, hopping forward/backward through frames or segments, "
        "and on-demand frame extraction (single or windowed) with automatic MinIO upload."
    ),
    purpose="Enable agents to spatially and temporally explore videos like a human reviewer — "
            "jump to moments, read captions/transcripts, pull surrounding context, "
            "and extract precise frames for inspection or downstream use.",
    
    typical_inputs_from=["Semantic Searcher", "Worker Result Inspector"],
)

ORCHESTRATOR_PLANNER_SPEC = BundleRole(
    name="Task Planner",
    description="Responsible for analysing the user query and breaking it into clear, parallelisable, non-overlapping sub-tasks for evidence workers.",
    purpose="Convert a complex user request into 2-6 precise, actionable worker objectives covering different modalities (person, object, action, event, scene, quote, timeline, etc.)."
)

VIDEO_EVIDENCE_WORKER_BUNDLE = BundleSpec(
    name="Video Evidence Worker",
    description="Worker strategy: find → verify → persist visual/textual evidence via iterative search, navigation, and transcript grounding.",

    roles={
        BundleRoles.QUERY_ANALYZER: QUERY_ANALYZER_SPEC,
        BundleRoles.SEMANTIC_SEARCHER: SEMANTIC_SEARCHER_SPEC,
        BundleRoles.TRANSCRIPT_ANALYZER: TRANSCRIPT_ANALYZER_SPEC,
        BundleRoles.WORKER_RESULT_INSPECTOR: WORKER_RESULT_INSPECTOR_SPEC,
        BundleRoles.VIDEO_NAVIGATOR: VIDEO_NAVIGATOR,
        BundleRoles.WORKER_EVIDENCE_MANAGER: WORKER_EVIDENCE_MANAGER_SPEC,
    },

    workflow_narrative="""
    **Worker Playbook (loop until done)**

    1. Enhance query → `{QUERY_ANALYZER}`
    2. Search (visual/caption/event/multimodal) → `{SEMANTIC_SEARCHER}`
    3. Inspect results & stats → `{WORKER_RESULT_INSPECTOR}`
    4. If promising → navigate video, hop segments/frames, extract frames → `{VIDEO_NAVIGATOR}`
    5. Always verify with surrounding transcript → `{TRANSCRIPT_ANALYZER}`
    6. When confident:
    • Persist evidence (score 7-10 + claims) → `{WORKER_EVIDENCE_MANAGER}`
    • Final call with summary → finish
    7. If nothing found after retries → `("No evidence found")` → finish

    Never guess. Always inspect before persisting.
    """.strip()
)


VIDEO_EVIDENCE_ORCHESTRATOR_BUNDLE = BundleSpec(
    name="Video Evidence Orchestrator",
    description="Top-level orchestrator that plans, delegates, monitors, synthesizes, and delivers final reports across evidence workers.",

    roles={
        BundleRoles.ORCHESTRATOR_PLANNER: ORCHESTRATOR_PLANNER_SPEC,
        BundleRoles.ORCHESTRATOR_WORK_INSPECTOR: ORCHESTRATOR_WORK_INSPECTOR_SPEC,
        BundleRoles.ORCHESTRATOR_EVIDENCE_MANAGER: ORCHESTRATOR_EVIDENCE_MANAGER_SPEC,
    },
    

    workflow_narrative="""
    **Orchestrator Playbook**

    1. **Plan**  
    Use your reasoning + `{ORCHESTRATOR_PLANNER}` mindset:  
    Break the user query into 2-6 clear sub-tasks (e.g. "person in red shirt", "gunshot sound", "license plate", "angry argument").

    2. **Delegate**  
    Launch parallel workers with exact sub-objectives.

    3. **Monitor**  
    While workers run → use `{ORCHESTRATOR_WORK_INSPECTOR}` to:  
    • View any worker's current evidence upon finish  
    • Read latest round summaries

    4. **Synthesize**  
    When all workers finish → use `{ORCHESTRATOR_EVIDENCE_MANAGER}` to:  
    • Update per-video context as needed  
    • Write final detailed report (process + evidence + conclusions)

    5. Never redo searches. Never hallucinate.

    6. If task can't be resolve, try to adjust the plan, and then rerun. Give up if nothing is reached.

    Final tool call **must**  from `{ORCHESTRATOR_EVIDENCE_MANAGER}`.
    """.strip()
)

