"""
plan_cache/config.py

PlanCacheConfig dataclass and preset factory functions.

Factory functions create a fully configured (PlanCache + Qdrant) bundle for
each of the four retrieval modes:

    get_exact_config()     - keyword hash → O(1) dict / Qdrant payload filter
    get_embedding_config() - dense vector similarity (needs an Embedder)
    get_bm25_config()      - BM25 keyword search via Qdrant sparse vectors
    get_hybrid_config()    - RRF fusion of dense + BM25
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from agno.learn.config import LearningMode

if TYPE_CHECKING:
    from agno.db.base import AsyncBaseDb, BaseDb
    from agno.models.base import Model
    from agno.vectordb.qdrant import Qdrant
    from agno.knowledge.embedder import Embedder
    from agno.vectordb.search import SearchType
    from agno.vectordb.distance import Distance

from .components import PlanCache, RetrievalMode


@dataclass
class PlanCacheConfig:
    """
    Configuration for PlanCacheStore.

    Parameters
    ----------
    plan_cache:
        Pre-built PlanCache instance (use a factory function below).
    retrieval_mode:
        Default retrieval strategy.  Kept in sync with plan_cache.mode.
    mode:
        LearningMode that controls *when* the learning store is activated.
    db:
        Optional relational DB backend (agno BaseDb / AsyncBaseDb).
    model:
        Lightweight model used for keyword extraction and cache generation.
    schema:
        Override the PlanTemplate schema (advanced use).
    candidate_threshold:
        Minimum similarity score for a candidate to be returned (0–1).
    quality_threshold:
        Minimum hit-rate before a template is eligible for eviction.
    min_attempts_for_eviction:
        Minimum total uses before quality-based eviction is considered.
    top_k_candidates:
        Maximum number of candidates returned by a lookup.
    enable_adaptation_tools:
        If True, get_tools() returns adaptation helpers for the agent.
    blueprint_store:
        Optional AgentBlueprintStore for hydrating blueprints alongside plans.
    instruction / additional_instructions:
        Optional system-prompt injections for the LearningMachine.
    """

    plan_cache:        "PlanCache"
    retrieval_mode:    RetrievalMode  = RetrievalMode.EXACT
    mode:              LearningMode   = LearningMode.ALWAYS

    db:     "Optional[BaseDb | AsyncBaseDb]" = None
    model:  "Optional[Model]"               = None
    schema: Any                             = None

    candidate_threshold:       float = 0.5
    quality_threshold:         float = 0.3
    min_attempts_for_eviction: int   = 5
    top_k_candidates:          int   = 5
    enable_adaptation_tools:   bool  = True

    blueprint_store: "Any | None" = None   

    instruction:             Optional[str] = None
    additional_instructions: Optional[str] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.candidate_threshold <= 1.0):
            raise ValueError(
                f"candidate_threshold must be between 0.0 and 1.0, "
                f"got {self.candidate_threshold}"
            )
        if not (0.0 <= self.quality_threshold <= 1.0):
            raise ValueError(
                f"quality_threshold must be between 0.0 and 1.0, "
                f"got {self.quality_threshold}"
            )
        if self.plan_cache is not None:
            self.retrieval_mode = self.plan_cache.mode

@dataclass
class AgentBlueprintConfig:
    """Configuration for AgentBlueprintStore."""

    blueprint_cache: Any  # AgentBlueprintCache

    # Retrieval settings
    retrieval_mode:     RetrievalMode = RetrievalMode.EXACT
    candidate_threshold: float        = 0.6
    top_k_blueprints:   int           = 3

    # Quality control
    quality_threshold:          float = 0.2
    min_attempts_for_eviction:  int   = 5

    # Feature flags
    enable_reconstruction_tool:     bool = True
    enable_comparison_tool:         bool = True
    include_limitations_in_context: bool = True

    # Optional SCOPE-style prompt evolver
    evolver: Any = None  # Optional[PromptEvolver]

    def __post_init__(self) -> None:
        if not (0.0 <= self.candidate_threshold <= 1.0):
            raise ValueError(
                f"candidate_threshold must be 0–1, got {self.candidate_threshold}"
            )
        # Guard: blueprint_cache may be None during construction in some test
        # scenarios — only sync retrieval_mode when it is actually set.
        if self.blueprint_cache is not None:
            self.retrieval_mode = self.blueprint_cache.mode


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def get_embedding_config(
    collection_name: str,
    qdrant_url: str,
    embedder: "Embedder",
    embedding_score_threshold: float = 0.7,
    candidate_limits: int = 10,
    quality_threshold: float = 0.3,
    min_attempts_for_eviction: int = 5,
    **kwargs: Any,
) -> PlanCacheConfig:
    """Dense-vector semantic search via Qdrant cosine similarity."""
    from agno.vectordb.qdrant import Qdrant
    from agno.vectordb.search import SearchType
    from agno.vectordb.distance import Distance

    qdrant_db = Qdrant(
        collection=collection_name,
        url=qdrant_url,
        embedder=embedder,
        search_type=SearchType.vector,
        distance=Distance.cosine,
    )
    plan_cache = PlanCache(
        vector_db=qdrant_db,
        quality_threshold=quality_threshold,
        min_attempts_for_eviction=min_attempts_for_eviction,
        mode=RetrievalMode.EMBEDDING,
    )
    return PlanCacheConfig(
        plan_cache=plan_cache,
        candidate_threshold=embedding_score_threshold,
        top_k_candidates=candidate_limits,
        quality_threshold=quality_threshold,
        min_attempts_for_eviction=min_attempts_for_eviction,
        **kwargs,
    )


def get_bm25_config(
    collection_name: str,
    qdrant_url: str,
    keyword_score_threshold: float = 0.3,
    candidate_limits: int = 10,
    quality_threshold: float = 0.3,
    min_attempts_for_eviction: int = 5,
    **kwargs: Any,
) -> PlanCacheConfig:
    """BM25 keyword search using Qdrant sparse vectors (fastembed)."""
    from agno.vectordb.qdrant import Qdrant
    from agno.vectordb.search import SearchType

    qdrant_db = Qdrant(
        collection=collection_name,
        url=qdrant_url,
        search_type=SearchType.keyword,
        sparse_vector_name="sparse",
        fastembed_kwargs={"model_name": "Qdrant/bm25"},
    )
    plan_cache = PlanCache(
        vector_db=qdrant_db,
        quality_threshold=quality_threshold,
        min_attempts_for_eviction=min_attempts_for_eviction,
        mode=RetrievalMode.FUZZY,
    )
    return PlanCacheConfig(
        plan_cache=plan_cache,
        candidate_threshold=keyword_score_threshold,
        top_k_candidates=candidate_limits,
        quality_threshold=quality_threshold,
        min_attempts_for_eviction=min_attempts_for_eviction,
        **kwargs,
    )


def get_hybrid_config(
    collection_name: str,
    qdrant_url: str,
    embedder: "Embedder",
    hybrid_score_threshold: float = 0.05,
    candidate_limits: int = 10,
    quality_threshold: float = 0.3,
    min_attempts_for_eviction: int = 5,
    **kwargs: Any,
) -> PlanCacheConfig:
    """Hybrid search: dense + BM25 fused via Qdrant's native RRF."""
    from agno.vectordb.qdrant import Qdrant
    from agno.vectordb.search import SearchType

    qdrant_db = Qdrant(
        collection=collection_name,
        url=qdrant_url,
        embedder=embedder,
        search_type=SearchType.hybrid,
        sparse_vector_name="sparse",
        hybrid_fusion_strategy="rrf",
    )
    plan_cache = PlanCache(
        vector_db=qdrant_db,
        quality_threshold=quality_threshold,
        min_attempts_for_eviction=min_attempts_for_eviction,
        mode=RetrievalMode.HYBRID,
    )
    return PlanCacheConfig(
        plan_cache=plan_cache,
        candidate_threshold=hybrid_score_threshold,
        top_k_candidates=candidate_limits,
        quality_threshold=quality_threshold,
        min_attempts_for_eviction=min_attempts_for_eviction,
        **kwargs,
    )


def get_exact_config(
    collection_name: str,
    qdrant_url: str,
    candidate_limits: int = 1,
    quality_threshold: float = 0.3,
    min_attempts_for_eviction: int = 5,
    **kwargs: Any,
) -> PlanCacheConfig:
    """
    Exact keyword matching.

    Primary lookups use the in-memory dict (O(1)).  Qdrant is used for
    cross-session persistence only; no embedder is required.
    candidate_threshold is fixed at 1.0 (exact match = perfect confidence).
    """
    from agno.vectordb.qdrant import Qdrant
    from agno.vectordb.search import SearchType

    qdrant_db = Qdrant(
        collection=collection_name,
        url=qdrant_url,
        search_type=SearchType.vector,
    )
    plan_cache = PlanCache(
        vector_db=qdrant_db,
        quality_threshold=quality_threshold,
        min_attempts_for_eviction=min_attempts_for_eviction,
        mode=RetrievalMode.EXACT,
    )
    return PlanCacheConfig(
        plan_cache=plan_cache,
        candidate_threshold=1.0,
        top_k_candidates=candidate_limits,
        quality_threshold=quality_threshold,
        min_attempts_for_eviction=min_attempts_for_eviction,
        **kwargs,
    )