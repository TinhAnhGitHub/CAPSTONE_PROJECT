# Toward Agentic Video Understanding: An Index‑Then‑Act Paradigm

## Abstract
Long‑form videos remain difficult to query with natural language due to their temporal scale, multimodal density, and diverse semantics. We present a system that transforms videos into a family of artifacts—segments, frames, captions, and embeddings—and exposes them as tools usable by planning agents. The approach follows an index‑then‑act paradigm: (i) a Prefect‑orchestrated ingestion pipeline produces structured, relational artifacts persisted to object storage and a vector database; (ii) an agent runtime binds those artifacts as typed tools for retrieval, navigation, and reasoning. We argue this separation improves controllability, transparency, and reuse. We describe the architecture, methods, and design trade‑offs, and we discuss limitations and future directions for agentic video understanding.

## 1. Introduction
Human queries over video often target “moments”—visually salient frames or semantically coherent segments. Conventional single‑shot prompting of large models struggles to ground such requests without explicit indices. Our system adopts a two‑phase strategy: first, it constructs multimodal indices from raw videos; second, it equips agents with a tool interface to those indices so they can plan, retrieve, and compose evidence in response to user commands. The central premise is that well‑formed artifacts and clear tool contracts enable robust, explainable agent behavior in complex, time‑varying media.

## 2. Background and Conceptual Framing
- Shot boundary detection (autoshot) partitions continuous streams into manageable, event‑like segments.
- Automatic speech recognition (ASR) turns audio into time‑aligned tokens, providing textual anchors for weakly visual queries.
- Multimodal captioning (image/segment) distills visual content into text that supports dense and sparse retrieval.
- Vector search (dense/sparse; hybrid ranking) provides scalable similarity search over frames and segments.
- Agentic planning over tool APIs (e.g., registry‑driven function tools) encourages modularity and controlled execution.

Our design combines these components under consistent storage and lineage contracts, then reifies them as agent tools.

## 3. System Overview
The system comprises two cooperating subsystems:

1) Ingestion substrate (orchestration):
   - A Prefect flow coordinates upload, autoshot, ASR, frame extraction, captioning, embedding, and vector persistence.
   - Artifacts are saved to MinIO (binaries/JSON), tracked in PostgreSQL (lineage), and indexed in Milvus (vectors).

2) Agentic tool surface:
   - A registry of typed tools (search, navigation, utility, prompts) exposes video instances to agents.
   - A factory binds dependencies (Milvus, MinIO, Postgres, external encoders, LLM) and enforces return‑type formatting.
   - A multi‑agent workflow (greeting → planning → orchestration → workers → consolidation) composes tool calls into answers.

This separation allows the same store of artifacts to support diverse agent strategies without re‑processing videos.

## 4. Method: Indexing Pipeline
Ingestion follows a staged, partly parallel workflow:

1) Video registration: persist originals and initialize progress tracking (produces `VideoArtifact`).
2) Parallel understanding: autoshot produces temporally bounded segments; ASR produces time‑aligned tokens (produces `AutoshotArtifact`, `ASRArtifact`).
3) Derived artifacts: extract representative frames; caption segments and images via an LLM service (produces `ImageArtifact`, `SegmentCaptionArtifact`, `ImageCaptionArtifact`).
4) Embeddings: generate vectors for images and captions; for segments, compute dense and sparse representations (produces `ImageEmbeddingArtifact`, `TextCaptionEmbeddingArtifact`, `TextCapSegmentEmbedArtifact`).
5) Vector persistence: insert batches into Milvus with per‑collection schemas to enable hybrid search.
6) Aggregation: write a manifest and human‑readable run summary as a Prefect artifact.

Artifacts are self‑describing via consistent object keys and fields (e.g., frame indices, timestamps, user bucket). Lineage is recorded so downstream tools can reconstruct parent context without recomputation.

## 5. Method: Tooling & Agent Composition
### 5.1 Tool registry and binding
Each tool function is registered with metadata (category, tags, dependencies). A factory binds declared dependencies—Milvus clients, storage, database, external encoders, or an LLM—so agents operate on safe, typed handles rather than raw clients. Return types are Pydantic models (e.g., `ImageObjectInterface`, `SegmentObjectInterface`) that the formatter converts into structured content blocks, enabling downstream LLM reasoning and UI rendering.

### 5.2 Tool taxonomy
- Search tools: visual text→image, caption text→image, multimodal hybrid (visual + caption dense + caption sparse), text→segment (event‑level) retrieval.
- Interaction/navigation tools: hop forward/backward among segments; list all segments of a video; step through images temporally.
- IO and time utilities: fetch images/segments from MinIO; convert between timecodes and frames; extract ad‑hoc frames by timestamp.
- Prompt tools: enhance visual or textual queries; caption a newly sampled image with optional ASR context.

### 5.3 Example composition patterns
- Query refinement → multimodal retrieval → segment hopping → ASR snippet extraction → evidence‑grounded answer.
- Visual similarity from a reference frame → neighborhood exploration → re‑captioning focused on user intent.
- Event retrieval across a subset of videos (`list_video_id`) with user‑scoped filtering and hybrid reranking.

## 6. Findings and Usage Patterns (Anecdotal)
Across pilot scenarios, we observe recurrent behaviors:
- Agents pivot modalities when signals disagree (e.g., fall back to ASR when visual matches are tenuous).
- Shorter segments with coherent ASR often yield higher‑precision answers than isolated frames.
- Hybrid rankers combining visual and caption features improve stability over purely dense methods.
- Lightweight navigation (hop N segments) reduces redundant retrieval and prompts more structured reasoning.

These patterns motivate maintaining both visual and textual indices and a small set of navigation primitives.

## 7. Design Considerations
- Controllability: declarative tools with explicit dependencies constrain agent actions and simplify auditing.
- Idempotence: artifact IDs and Milvus existence checks prevent duplicate ingestion and reduce cost.
- Observability: Prefect logs, progress trackers, and run summaries help diagnose bottlenecks and drift.
- Interoperability: S3‑style URIs, JSON payloads, and typed interfaces decouple producers and consumers.
- Cost/latency: batching and hybrid search keep retrieval cost predictable while preserving recall.

## 8. Limitations & Threats to Validity
- Domain shift: embeddings and captioners may underperform on out‑of‑distribution video genres.
- ASR sensitivity: noisy audio or non‑supported languages degrade downstream segment caption quality.
- Index freshness: ingestion is currently batch‑oriented; online/near‑real‑time updates remain future work.
- Vector governance: schema evolution and collection growth require careful migration and monitoring.
- Privacy and compliance: storing frames, transcripts, and captions necessitates tenant isolation and retention policies.

## 9. Future Work
- Online indexing and incremental re‑embedding to reflect new content rapidly.
- Cross‑video event graphs for storyline or character‑centric queries.
- Learned aggregation/reranking that adapts weights per query type.
- Safety filters and policy‑aware tools that redact sensitive frames or transcript spans.
- Memory‑augmented agents that cache intermediate tool outputs across sessions.

## 10. Conclusion
By separating ingestion (index) from reasoning (act), the system makes videos legible to agents through a compact vocabulary of artifacts and tools. This architecture supports grounded, controllable answers over complex, time‑varying media. We expect continued progress in multimodal encoders and agent planning to further compress the distance between user intent and precise video moments.

