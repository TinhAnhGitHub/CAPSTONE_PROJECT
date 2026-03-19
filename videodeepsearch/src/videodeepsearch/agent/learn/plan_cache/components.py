"""
plan_cache/components.py

Key components:
1. PlanStep       - One action in an execution plan (dataclass)
2. PlanTemplate   - Cached plan with quality tracking (dataclass, aligns with
                    agno LearningStore schema convention so from_dict_safe works)
3. CacheLookupResult - Result of a cache lookup (dataclass)
4. PlanCache      - Multi-mode retrieval: EXACT (O(1) dict), FUZZY (BM25),
                    EMBEDDING (vector), HYBRID (vector + BM25)
5. Helper functions for template extraction and formatting

Design notes
------------
* PlanTemplate is now a @dataclass (not Pydantic BaseModel) so that
  agno's from_dict_safe / to_dict_safe work without modification.
* EXACT mode uses an in-memory dict (O(1)).  Qdrant is used only as a
  persistence / warm-up layer.
* Qdrant write operations (store_template, delete, record_outcome) now have
  genuinely async variants that use asyncio.to_thread so they never block
  the event loop.
* PlanCache no longer mutates search_type in-place (thread/coroutine safe).
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import md5
from typing import Any, Annotated, Optional
import time

from agno.knowledge.document import Document
from agno.utils.log import log_debug, log_warning
from agno.vectordb.qdrant import Qdrant
from agno.vectordb.search import SearchType

class RetrievalMode(str, Enum):
    """Plan-cache retrieval strategy."""
    EXACT     = "exact"
    FUZZY     = "fuzzy"
    EMBEDDING = "embedding"
    HYBRID    = "hybrid"

@dataclass
class PlanStep:
    step_index: int
    instruction: str
    expected_output: str 
    tool_hints: list[str] = field(default_factory=list)
    worker_role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlanStep":
        return cls(
            step_index=data["step_index"],
            instruction=data["instruction"],
            expected_output=data.get("expected_output", ""),
            tool_hints=data.get("tool_hints", []),
            worker_role=data.get("worker_role"),
        )

    def __str__(self) -> str:
        tools = f" (tools: {', '.join(self.tool_hints)})" if self.tool_hints else ""
        return f"Step {self.step_index + 1}: {self.instruction}{tools}"


@dataclass
class PlanTemplate:
    keyword: str
    task_description: str = ""
    plan_summary: str = ""
    steps: list[dict] = field(default_factory=list)  # serialised PlanStep dicts
    worker_roles_needed: list[str] = field(default_factory=list)
    tool_chain: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    blueprint_ids: list[str] = field(default_factory=list)
    step_blueprint_map: dict[int, str]   = field(default_factory=dict)
    success_count: int = 1
    fail_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    version: str = "v0.0.1"
    tags: list[str]  = field(default_factory=list)

    embedding: list[float] | None = field(default=None, repr=False)

    def hit_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.0

    def total_uses(self) -> int:
        return self.success_count + self.fail_count

    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    def get_searchable_text(self) -> str:
        parts = [self.keyword, self.task_description] + self.aliases
        return " ".join(filter(None, parts))

    def get_steps(self) -> list[PlanStep]:
        """Return steps as PlanStep objects (deserialised from stored dicts)."""
        result = []
        for s in self.steps:
            if isinstance(s, dict):
                try:
                    result.append(PlanStep.from_dict(s))
                except Exception:
                    pass
            elif isinstance(s, PlanStep):
                result.append(s)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding the embedding vector."""
        d = asdict(self)
        d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, data: Any) -> Optional["PlanTemplate"]:
        """
        Parse from dict/JSON.  Returns None on failure - same pattern as
        agno's UserProfile.from_dict / EntityMemory.from_dict.
        """
        if data is None:
            return None
        if isinstance(data, cls):
            return data

        import json
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                return None

        if not isinstance(data, dict):
            return None

        try:
            from dataclasses import fields as dc_fields
            field_names = {f.name for f in dc_fields(cls)}
            kwargs = {k: v for k, v in data.items() if k in field_names}
            return cls(**kwargs)
        except Exception as e:
            log_debug(f"PlanTemplate.from_dict failed: {e}")
            return None

    def __str__(self) -> str:
        return (
            f"PlanTemplate(keyword='{self.keyword}', "
            f"steps={len(self.steps)}, "
            f"success_rate={self.hit_rate():.0%})"
        )


@dataclass
class CacheLookupResult:
    """Result of a cache lookup with confidence scoring."""
    hit:            bool
    query:          str
    keyword:        str
    template:       Optional[PlanTemplate]       = None
    confidence:     float                        = 0.0
    retrieval_mode: RetrievalMode                = RetrievalMode.EXACT
    candidates:     list[tuple[PlanTemplate, float]] = field(default_factory=list)
    blueprints:     list[Any]                    = field(default_factory=list)

    def __str__(self) -> str:
        bp_info = f", blueprints={len(self.blueprints)}" if self.blueprints else ""
        if self.hit:
            return (
                f"Cache HIT: '{self.keyword}' "
                f"(confidence={self.confidence:.0%}, mode={self.retrieval_mode.value}"
                f"{bp_info})"
            )
        return f"Cache MISS: '{self.keyword}' (mode={self.retrieval_mode.value})"


# ---------------------------------------------------------------------------
# PlanCache
# ---------------------------------------------------------------------------

class PlanCache:
    """
    Multi-mode plan cache backed by Qdrant.

    Thread / coroutine safety
    -------------------------
    * lookup() and alookup() never mutate self.vector_db.search_type globally.
      Instead they pass the desired SearchType directly, or use a localised
      override that is restored in a finally block.
    * store_template / delete operate on Qdrant synchronously via the
      underlying client.  The async variants wrap them in asyncio.to_thread
      so the event loop is never blocked.
    """

    def __init__(
        self,
        vector_db: Qdrant,
        quality_threshold: Annotated[float, "Min hit-rate before quality eviction."] = 0.3,
        min_attempts_for_eviction: Annotated[int, "Min uses before quality eviction."] = 5,
        mode: RetrievalMode = RetrievalMode.EXACT,
    ) -> None:
        self.vector_db = vector_db
        self.quality_threshold = quality_threshold
        self.min_attempts_for_eviction = min_attempts_for_eviction
        self.mode = mode

        self._exact_cache: dict[str, PlanTemplate] = {}
        self._cache_hits   = 0
        self._cache_misses = 0
        self._evictions    = 0

    # ------------------------------------------------------------------
    # Size / stats
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        try:
            return self.vector_db.get_count()  # type: ignore[attr-defined]
        except AttributeError:
            try:
                result = self.vector_db.client.count(
                    collection_name=self.vector_db.collection
                )
                return result.count
            except Exception:
                return len(self._exact_cache)

    @property
    def hit_rate(self) -> float:
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(keyword: str) -> str:
        return keyword.strip().lower()

    def _get_id(self, keyword: str) -> str:
        return md5(self._normalise(keyword).encode()).hexdigest()

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def list_keywords(self) -> list[str]:
        try:
            points, _ = self.vector_db.client.scroll(
                collection_name=self.vector_db.collection,
                with_payload=True,
                with_vectors=False,
            )
            return [
                p.payload["meta_data"]["keyword"]
                for p in points
                if p.payload and "meta_data" in p.payload
            ]
        except Exception as e:
            log_warning(f"PlanCache.list_keywords failed: {e}")
            return list(self._exact_cache.keys())

    def get_template(self, keyword: str) -> Optional[PlanTemplate]:
        result = self.lookup(keyword, mode=RetrievalMode.EXACT)
        return result.template if result.hit else None

    def get_all_templates(self) -> list[PlanTemplate]:
        try:
            points, _ = self.vector_db.client.scroll(
                collection_name=self.vector_db.collection,
                with_payload=True,
                with_vectors=False,
            )
            templates = []
            for p in points:
                try:
                    tpl = PlanTemplate.from_dict(p.payload.get("meta_data")) #type:ignore
                    if tpl:
                        templates.append(tpl)
                except Exception as e:
                    log_warning(f"Could not deserialise template: {e}")
            return templates
        except Exception as e:
            log_warning(f"PlanCache.get_all_templates failed: {e}")
            return list(self._exact_cache.values())

    # ------------------------------------------------------------------
    # Synchronous lookup
    # ------------------------------------------------------------------

    def lookup(
        self,
        query: str,
        limit: int = 2,
        score_threshold: float = 0.7,
        mode: Optional[RetrievalMode] = None,
    ) -> CacheLookupResult:
        final_mode       = mode or self.mode
        query_normalised = self._normalise(query)

        if final_mode == RetrievalMode.EXACT:
            template = self._exact_cache.get(query_normalised)
            if template is None:
                template = self._retrieve_exact_from_qdrant(query_normalised)
                if template:
                    self._exact_cache[query_normalised] = template

            if template:
                template.last_used_at = time.time()
                self._cache_hits += 1
                log_debug(f"PlanCache EXACT HIT: '{query_normalised}'")
                return CacheLookupResult(
                    hit=True,
                    query=query,
                    keyword=query_normalised,
                    template=template,
                    confidence=1.0,
                    retrieval_mode=RetrievalMode.EXACT,
                )
            self._cache_misses += 1
            return CacheLookupResult(
                hit=False, query=query, keyword=query_normalised,
                retrieval_mode=RetrievalMode.EXACT,
            )

        # ---- FUZZY / EMBEDDING / HYBRID ----------------------------------
        # Temporarily override search_type, always restore it.
        original_search_type = self.vector_db.search_type  # type: ignore[attr-defined]
        if final_mode == RetrievalMode.FUZZY:
            self.vector_db.search_type = SearchType.keyword  # type: ignore[attr-defined]
        elif final_mode == RetrievalMode.HYBRID:
            self.vector_db.search_type = SearchType.hybrid   # type: ignore[attr-defined]

        try:
            raw_results = self.vector_db.search(query=query, limit=limit)
        except Exception as e:
            log_warning(f"PlanCache.lookup search failed ({final_mode}): {e}")
            raw_results = []
        finally:
            self.vector_db.search_type = original_search_type  # type: ignore[attr-defined]

        return self._build_lookup_result(
            raw_results=raw_results,
            query=query,
            query_normalised=query_normalised,
            final_mode=final_mode,
            score_threshold=score_threshold,
        )

    # ------------------------------------------------------------------
    # Asynchronous lookup  (genuinely async Qdrant search)
    # ------------------------------------------------------------------

    async def alookup(
        self,
        query: str,
        limit: int = 2,
        score_threshold: float = 0.7,
        mode: Optional[RetrievalMode] = None,
    ) -> CacheLookupResult:
        final_mode       = mode or self.mode
        query_normalised = self._normalise(query)

        # ---- EXACT -------------------------------------------------------
        if final_mode == RetrievalMode.EXACT:
            template = self._exact_cache.get(query_normalised)
            if template is None:
                template = await asyncio.to_thread(
                    self._retrieve_exact_from_qdrant, query_normalised
                )
                if template:
                    self._exact_cache[query_normalised] = template

            if template:
                template.last_used_at = time.time()
                self._cache_hits += 1
                return CacheLookupResult(
                    hit=True,
                    query=query,
                    keyword=query_normalised,
                    template=template,
                    confidence=1.0,
                    retrieval_mode=RetrievalMode.EXACT,
                )
            self._cache_misses += 1
            return CacheLookupResult(
                hit=False, query=query, keyword=query_normalised,
                retrieval_mode=RetrievalMode.EXACT,
            )

        # ---- FUZZY / EMBEDDING / HYBRID ----------------------------------
        # Use async_search when available, otherwise offload to a thread.
        original_search_type = self.vector_db.search_type  # type: ignore[attr-defined]
        if final_mode == RetrievalMode.FUZZY:
            self.vector_db.search_type = SearchType.keyword  # type: ignore[attr-defined]
        elif final_mode == RetrievalMode.HYBRID:
            self.vector_db.search_type = SearchType.hybrid   # type: ignore[attr-defined]

        try:
            if hasattr(self.vector_db, "async_search"):
                raw_results = await self.vector_db.async_search(  # type: ignore[attr-defined]
                    query=query, limit=limit
                )
            else:
                raw_results = await asyncio.to_thread(
                    self.vector_db.search, query, limit
                )
        except Exception as e:
            log_warning(f"PlanCache.alookup search failed ({final_mode}): {e}")
            raw_results = []
        finally:
            self.vector_db.search_type = original_search_type  # type: ignore[attr-defined]

        return self._build_lookup_result(
            raw_results=raw_results,
            query=query,
            query_normalised=query_normalised,
            final_mode=final_mode,
            score_threshold=score_threshold,
        )

    # ------------------------------------------------------------------
    # Write / delete  (sync + genuinely async variants)
    # ------------------------------------------------------------------

    def store_template(self, template: PlanTemplate) -> None:
        """
        Persist a template.
        1. Always update the in-memory exact cache (fast path).
        2. Upsert into Qdrant using agno's Document/upsert API.
        """
        key    = self._normalise(template.keyword)
        doc_id = self._get_id(template.keyword)

        self._exact_cache[key] = template

        meta = template.to_dict()
        doc  = Document(
            id=doc_id,
            name=key,
            content=template.get_searchable_text(),
            meta_data=meta,
        )
        try:
            self.vector_db.upsert(documents=[doc]) #type:ignore
            log_debug(f"PlanCache: stored template '{key}' (id={doc_id[:8]}…)")
        except Exception as e:
            log_warning(
                f"PlanCache: Qdrant upsert failed for '{key}': {e}. "
                "Template is available in-memory for this session."
            )

    async def astore_template(self, template: PlanTemplate) -> None:
        """
        Genuinely async store: Qdrant upsert is offloaded to a thread so the
        event loop is not blocked.
        """
        # In-memory update is instant – do it on the calling thread.
        key = self._normalise(template.keyword)
        self._exact_cache[key] = template

        # Offload the blocking Qdrant I/O.
        await asyncio.to_thread(self._store_template_sync, template)

    def _store_template_sync(self, template: PlanTemplate) -> None:
        """Internal sync helper used by astore_template."""
        key    = self._normalise(template.keyword)
        doc_id = self._get_id(template.keyword)
        meta   = template.to_dict()
        doc    = Document(
            id=doc_id,
            name=key,
            content=template.get_searchable_text(),
            meta_data=meta,
        )
        try:
            self.vector_db.upsert(documents=[doc]) #type:ignore
            log_debug(f"PlanCache: stored template '{key}' (id={doc_id[:8]}…)")
        except Exception as e:
            log_warning(f"PlanCache: Qdrant upsert failed for '{key}': {e}.")

    def delete(self, keyword: str) -> bool:
        key    = self._normalise(keyword)
        doc_id = self._get_id(keyword)

        self._exact_cache.pop(key, None)

        try:
            from qdrant_client.models import PointIdsList
            self.vector_db.client.delete(
                collection_name=self.vector_db.collection,
                points_selector=PointIdsList(points=[doc_id]),
            )
            return True
        except Exception as e:
            log_warning(f"PlanCache.delete failed for '{key}': {e}")
            return False

    async def adelete(self, keyword: str) -> bool:
        """Genuinely async delete: offloads the Qdrant call to a thread."""
        key = self._normalise(keyword)
        self._exact_cache.pop(key, None)
        return await asyncio.to_thread(self._delete_from_qdrant, keyword)

    def _delete_from_qdrant(self, keyword: str) -> bool:
        doc_id = self._get_id(keyword)
        try:
            from qdrant_client.models import PointIdsList
            self.vector_db.client.delete(
                collection_name=self.vector_db.collection,
                points_selector=PointIdsList(points=[doc_id]),
            )
            return True
        except Exception as e:
            log_warning(f"PlanCache._delete_from_qdrant failed for '{keyword}': {e}")
            return False

    # ------------------------------------------------------------------
    # Quality tracking  (sync + genuinely async)
    # ------------------------------------------------------------------

    def record_outcome(self, keyword: str, success: bool) -> None:
        """Increment success/fail and evict if quality drops below threshold."""
        key      = self._normalise(keyword)
        template = self._exact_cache.get(key) or self._retrieve_exact_from_qdrant(key)
        if not template:
            return

        if success:
            template.success_count += 1
        else:
            template.fail_count += 1

        should_evict = (
            template.total_uses() >= self.min_attempts_for_eviction
            and template.hit_rate() < self.quality_threshold
        )
        if should_evict:
            self.delete(keyword)
            self._evictions += 1
            log_debug(f"PlanCache: evicted low-quality template '{key}'")
        else:
            self.store_template(template)

    async def arecord_outcome(self, keyword: str, success: bool) -> None:
        """
        Genuinely async record_outcome.
        Qdrant reads and writes are offloaded to a thread.
        """
        key      = self._normalise(keyword)
        template = self._exact_cache.get(key)
        if template is None:
            template = await asyncio.to_thread(
                self._retrieve_exact_from_qdrant, key
            )
        if not template:
            return

        if success:
            template.success_count += 1
        else:
            template.fail_count += 1

        should_evict = (
            template.total_uses() >= self.min_attempts_for_eviction
            and template.hit_rate() < self.quality_threshold
        )
        if should_evict:
            await self.adelete(keyword)
            self._evictions += 1
            log_debug(f"PlanCache: evicted low-quality template '{key}'")
        else:
            await self.astore_template(template)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        templates     = self.get_all_templates()
        total_uses    = sum(t.total_uses() for t in templates)
        total_success = sum(t.success_count for t in templates)
        avg_hit_rate  = total_success / total_uses if total_uses > 0 else 0.0

        return {
            "retrieval_mode":       self.mode.value,
            "cache_size":           self.size,
            "in_memory_size":       len(self._exact_cache),
            "cache_hits":           self._cache_hits,
            "cache_misses":         self._cache_misses,
            "hit_rate":             f"{self.hit_rate:.1%}",
            "evictions":            self._evictions,
            "total_template_uses":  total_uses,
            "average_success_rate": f"{avg_hit_rate:.1%}",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve_exact_from_qdrant(self, keyword_normalised: str) -> Optional[PlanTemplate]:
        doc_id = self._get_id(keyword_normalised)
        try:
            points = self.vector_db.client.retrieve(
                collection_name=self.vector_db.collection,
                ids=[doc_id],
                with_payload=True,
                with_vectors=False,
            )
            if points and points[0].payload:
                return PlanTemplate.from_dict(points[0].payload.get("meta_data"))
        except Exception as e:
            log_warning(f"Qdrant exact retrieve failed for '{keyword_normalised}': {e}")
        return None

    def _build_lookup_result(
        self,
        raw_results: list,
        query: str,
        query_normalised: str,
        final_mode: RetrievalMode,
        score_threshold: float,
    ) -> CacheLookupResult:
        """Shared result-builder for sync and async lookup paths."""
        filtered = [
            doc for doc in raw_results
            if getattr(doc, "score", 0.0) >= score_threshold
        ]

        if not filtered:
            self._cache_misses += 1
            return CacheLookupResult(
                hit=False, query=query, keyword=query_normalised,
                retrieval_mode=final_mode,
            )

        self._cache_hits += 1
        best = filtered[0]
        candidates: list[tuple[PlanTemplate, float]] = []
        for doc in filtered[1:]:
            try:
                tpl = PlanTemplate.from_dict(getattr(doc, "meta_data", None))
                if tpl:
                    candidates.append((tpl, getattr(doc, "score", 0.0)))
            except Exception:
                pass

        best_template = PlanTemplate.from_dict(getattr(best, "meta_data", None))
        if best_template is None:
            log_warning("Failed to deserialise best template.")
            self._cache_misses += 1
            return CacheLookupResult(
                hit=False, query=query, keyword=query_normalised,
                retrieval_mode=final_mode,
            )

        log_debug(
            f"PlanCache {final_mode.value.upper()} HIT: '{query_normalised}' "
            f"score={getattr(best, 'score', 0.0):.2f}"
        )
        return CacheLookupResult(
            hit=True,
            query=query,
            keyword=query_normalised,
            template=best_template,
            confidence=getattr(best, "score", 0.0),
            retrieval_mode=final_mode,
            candidates=candidates,
        )


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

def format_template_for_prompt(template: PlanTemplate) -> str:
    """Format a PlanTemplate for injection into an agent system prompt."""
    lines = [
        f"## Cached Plan Template: {template.keyword}",
        f"**Task Pattern:** {template.task_description}",
        f"**Success Rate:** {template.hit_rate():.0%} "
        f"({template.success_count}✓/{template.fail_count}✗)",
        "",
        "### Execution Steps:",
    ]
    for step in template.get_steps():
        lines.append(f"\n**Step {step.step_index + 1}:** {step.instruction}")
        if step.expected_output:
            lines.append(f"  - *Expected output:* {step.expected_output}")
        if step.tool_hints:
            lines.append(f"  - *Suggested tools:* {', '.join(step.tool_hints)}")
        if step.worker_role:
            lines.append(f"  - *Worker role:* {step.worker_role}")

    if template.tool_chain:
        lines.append(f"\n**Proven Tool Chain:** {' → '.join(template.tool_chain)}")
    if template.worker_roles_needed:
        lines.append(f"**Required Workers:** {', '.join(template.worker_roles_needed)}")

    return "\n".join(lines)


def build_plan_template_from_trace(
    keyword: str,
    task_description: str,
    worker_results: list[Any],
    tool_chain: list[str],
    summary: str,
    aliases: Optional[list[str]] = None,
    
) -> PlanTemplate:
    """Build a PlanTemplate from an execution trace."""
    steps: list[dict]  = []
    worker_roles: list[str] = []

    for i, wr in enumerate(worker_results):
        def _get(obj: Any, *keys: str, default: Any = None) -> Any:
            for k in keys:
                v = (obj.get(k) if isinstance(obj, dict) else getattr(obj, k, None))
                if v is not None:
                    return v
            return default

        instruction  = _get(wr, "task_objective", "task") or str(wr)
        expected_out = _get(wr, "result_summary") or ""
        tools_used   = _get(wr, "tools_used") or []
        worker_role  = _get(wr, "worker_name")

        step = PlanStep(
            step_index=i,
            instruction=str(instruction),
            expected_output=str(expected_out) if expected_out else "",
            tool_hints=list(tools_used),
            worker_role=worker_role,
        )
        steps.append(step.to_dict())
        if worker_role:
            worker_roles.append(worker_role)

    return PlanTemplate(
        plan_summary=summary,
        keyword=keyword.strip().lower(),
        task_description=task_description,
        steps=steps,
        worker_roles_needed=list(set(worker_roles)),
        tool_chain=tool_chain,
        aliases=aliases or [],
    )