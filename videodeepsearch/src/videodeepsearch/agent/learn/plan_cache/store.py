"""
plan_cache/store.py

PlanCacheStore - a custom agno LearningStore implementing Agentic Plan Caching
(APC, Zhang et al., NeurIPS 2025).

Integration with agno's LearningMachine
----------------------------------------
recall()        Keyword extraction → cache lookup → returns CacheLookupResult
build_context() Formats the retrieved template into a system-prompt snippet
get_tools()     Exposes plan-adaptation helpers to the agent (optional)
process()       On execution completion: record outcome + extract / cache template
arecall / aprocess / aget_tools  - async mirrors of the above

Key design decisions vs. original
-----------------------------------
1. **No mutable per-request instance state.**
   The original stored current_keyword, current_template, cache_hit, etc. as
   instance attributes.  This causes race conditions when the same store is
   shared across concurrent async requests (common in agno teams / workflows).

   The fix: recall() returns a CacheLookupResult that carries everything the
   caller needs.  build_context() and get_tools() accept the result as *data*
   (matching the LearningStore protocol signature) instead of reading instance
   fields.  process() receives the keyword / template explicitly via **kwargs
   so callers that have the result from recall() can pass it through.

2. **process() accepts execution_trace via **kwargs.**
   agno's LearningMachine.process() calls store.process(**context) where
   context = {messages, user_id, session_id, ...}.  worker_results / tool_chain
   are NOT in that context dict, so they must come from the caller explicitly
   via kwargs (or via a direct call bypassing LearningMachine).  We document
   this clearly and handle the absent-kwarg case gracefully.

3. **Genuinely async Qdrant calls.**
   arecord_outcome / astore_template / adelete all delegate to asyncio.to_thread
   so the event loop is never blocked (see components.py).

4. **Tool factories receive a snapshot of the lookup result.**
   Closures capture the CacheLookupResult passed in at get_tools() time, not
   stale instance state.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from os import getenv
from textwrap import dedent
from typing import Any, Optional, cast
from collections.abc import Callable

from agno.db.base import AsyncBaseDb, BaseDb
from agno.learn.stores.protocol import LearningStore
from agno.models.base import Model
from agno.utils.log import (
    log_debug,
    log_warning,
    set_log_level_to_debug,
    set_log_level_to_info,
)


from agno.db.base import AsyncBaseDb, BaseDb
from agno.models.message import Message
from agno.tools.function import Function

from .components import (
    CacheLookupResult,
    PlanCache,
    PlanTemplate,
    RetrievalMode,
    build_plan_template_from_trace,
    format_template_for_prompt,
)
from .config import PlanCacheConfig

try:
    from agent_blueprint_cache.store import AgentBlueprintStore as _AgentBlueprintStore
    from agent_blueprint_cache.components import AgentBlueprint as _AgentBlueprint
    _BLUEPRINT_AVAILABLE = True
except ImportError:
    _BLUEPRINT_AVAILABLE = False


@dataclass
class PlanCacheStore(LearningStore):
    """
    agno LearningStore that implements Agentic Plan Caching.

    Usage (minimal)
    ---------------
    >>> config = get_exact_config("my_collection", "http://localhost:6333",
    ...                           model=OpenAIChat(id="gpt-4o-mini"))
    >>> store  = PlanCacheStore(config=config)
    >>> # Attach to an Agent or Team via learning=[store]

    Concurrency note
    ----------------
    This store holds NO per-request mutable state.  It is safe to share a
    single PlanCacheStore instance across concurrent async runs.
    The only mutable field is _template_stored which is set during process()
    and immediately consumed by was_updated - callers should not rely on it
    across concurrent calls.
    """

    config: PlanCacheConfig
    debug_mode: bool = False

    _template_stored: bool = field(default=False, init=False)
    _schema: Any  = field(default=None,  init=False)

    def __post_init__(self) -> None:
        self._schema = self.config.schema or PlanTemplate
        self.set_log_level()

    @property
    def learning_type(self) -> str:
        return "plan_cache"

    @property
    def schema(self) -> Any:
        return self._schema

    @property
    def was_updated(self) -> bool:
        """True if a new template was stored during the last process() call."""
        return self._template_stored

    @property
    def db(self) -> BaseDb | AsyncBaseDb | None:
        return self.config.db

    @property
    def model(self) -> Model | None:
        return self.config.model

    @property
    def blueprint_store(self) -> Any | None:
        return self.config.blueprint_store

    @property
    def plan_cache(self) -> Optional[PlanCache]:
        return self.config.plan_cache

    def recall(
        self,
        session_id: str | None = None,
        query: str | None = None,
        retrieval_mode: RetrievalMode | None = None,
        **kwargs: Any,
    ) -> Optional[CacheLookupResult]:
        """
        Synchronous cache lookup.

        Returns a CacheLookupResult that carries everything needed by
        build_context() and get_tools().  No mutable instance state is set.
        """
        keyword = kwargs.get("keywords") or kwargs.get("keyword")
        if not keyword and query:
            keyword = self._extract_keyword(query)
        keyword = keyword or query or ""

        effective_query = query or keyword

        result = self.config.plan_cache.lookup(
            query=keyword,
            limit=self.config.top_k_candidates,
            score_threshold=self.config.candidate_threshold,
            mode=retrieval_mode,
        )

        if result.hit and result.template and self.blueprint_store is not None:
            try:
                bp_result = self.blueprint_store.recall_for_plan(
                    keyword=result.keyword,
                    blueprint_ids=result.template.blueprint_ids or None,
                )
                if bp_result.hit:
                    result.blueprints = bp_result.blueprints
            except Exception as e:
                log_warning(f"PlanCacheStore.recall: blueprint hydration failed: {e}")

        log_debug(
            f"PlanCacheStore.recall: keyword='{keyword[:60]}', "
            f"hit={result.hit}, confidence={result.confidence:.2f}, "
            f"mode={result.retrieval_mode.value}, "
            f"candidates={len(result.candidates)}"
        )
        return result

    async def arecall(
        self,
        session_id: str | None = None,
        query: str | None = None,
        retrieval_mode: RetrievalMode | None = None,
        **kwargs: Any,
    ) -> Optional[CacheLookupResult]:
        """Async cache lookup."""
        keyword = kwargs.get("keywords") or kwargs.get("keyword")
        if not keyword and query:
            keyword = await self._aextract_keyword(query)
        keyword = keyword or query or ""

        result = await self.config.plan_cache.alookup(
            query=keyword,
            limit=self.config.top_k_candidates,
            score_threshold=self.config.candidate_threshold,
            mode=retrieval_mode,
        )

        if result.hit and result.template and self.blueprint_store is not None:
            try:
                bp_result = await self.blueprint_store.arecall_for_plan(
                    keyword=result.keyword,
                    blueprint_ids=result.template.blueprint_ids or None,
                )
                if bp_result.hit:
                    result.blueprints = bp_result.blueprints
            except Exception as e:
                log_warning(f"PlanCacheStore.arecall: blueprint hydration failed: {e}")

        log_debug(
            f"PlanCacheStore.arecall: keyword='{keyword[:60]}', "
            f"hit={result.hit}, confidence={result.confidence:.2f}, "
            f"mode={result.retrieval_mode.value}, "
            f"candidates={len(result.candidates)}"
        )
        return result

    def build_context(self, data: Optional[CacheLookupResult]) -> str:
        """
        Return a system-prompt snippet from the lookup result.
        Returns "" on a cache miss or None input.
        """
        if not data or not data.hit or not data.template:
            return ""

        template_text = format_template_for_prompt(data.template)

        mode_label = {
            RetrievalMode.EXACT: "This is an exact match from the cache.",
            RetrievalMode.FUZZY: f"This is a fuzzy match (confidence: {data.confidence:.0%}).",
            RetrievalMode.EMBEDDING: f"This is a semantic match (similarity: {data.confidence:.0%}).",
            RetrievalMode.HYBRID: f"This is a hybrid match (score: {data.confidence:.0%}).",
        }.get(data.retrieval_mode, "")

        alternatives = ""
        if data.candidates:
            alt_lines = [
                f"- {t.keyword} (score: {s:.0%})" for t, s in data.candidates[:3]
            ]
            alternatives = "\n\n**Alternative approaches:**\n" + "\n".join(alt_lines)

        confidence_hint = (
            "High confidence – follow closely."
            if data.confidence > 0.8
            else "Moderate confidence – adapt as needed."
        )

        plan_context = dedent(f"""\
            <cached_plan_template>
            You have solved similar tasks before. Here is a proven plan template:

            {template_text}

            {mode_label}
            {alternatives}

            <adaptation_guidelines>
            Adapt the template to the current context:
            1. Replace placeholders ([Entity], [Date], etc.) with actual values.
            2. Adjust step details to match the specific requirements.
            3. Maintain the overall flow and tool chain where possible.
            4. Skip irrelevant steps if needed.

            Confidence: {data.confidence:.0%} - {confidence_hint}
            </adaptation_guidelines>
            </cached_plan_template>\
        """)

        if data.blueprints and self.blueprint_store is not None:
            try:
                from agent_blueprint_cache.components import BlueprintLookupResult as _BLR
                bp_result = _BLR(
                    hit=True,
                    query=data.keyword,
                    keyword=data.keyword,
                    best=data.blueprints[0],
                    blueprints=data.blueprints,
                    confidence=data.confidence,
                    retrieval_mode=data.retrieval_mode,
                )
                blueprint_section = self.blueprint_store.build_context(bp_result)
                if blueprint_section:
                    plan_context = plan_context + "\n\n" + blueprint_section
            except Exception as e:
                log_warning(f"PlanCacheStore.build_context: blueprint section failed: {e}")

        return plan_context

    def process(
        self,
        messages: list[Any],
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        team_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Called after an agent run completes.

        Standard agno kwargs (user_id, session_id, namespace, etc.) are accepted
        and ignored.  Plan-cache-specific data is passed via extra kwargs:

            execution_trace : list  - worker result objects to build a template from
            success         : bool  - whether the run succeeded (default True)
            keyword         : str   - pre-extracted keyword (skips LLM extraction)
            tool_chain      : list[str]
            task_description: str
            aliases         : list[str]
            blueprints / agents / agent  - forwarded to blueprint_store if configured

        Why kwargs instead of explicit parameters?
        ------------------------------------------
        agno's LearningMachine.process() calls store.process(**context) where
        context only contains {messages, user_id, session_id, namespace, ...}.
        The plan-cache extras (execution_trace, etc.) must come from the
        application layer that calls process() directly.  Using **kwargs keeps
        the signature compatible with both call paths.
        """
        self._template_stored = False
        success        = kwargs.get("success", True)
        execution_trace = kwargs.get("execution_trace")
        keyword = kwargs.get("keyword")
        if keyword:
            self.config.plan_cache.record_outcome(keyword, success)

        if not success:
            return

        if execution_trace:
            self.extract_and_cache(
                messages=messages,
                execution_trace=execution_trace,
                session_id=session_id,
                user_id=user_id,
                agent_id=agent_id,
                team_id=team_id,
                keyword=keyword,
                tool_chain=kwargs.get("tool_chain"),
                task_description=kwargs.get("task_description"),
                aliases=kwargs.get("aliases"),
            )

        self._link_blueprints_to_template(
            messages=messages,
            keyword=keyword,
            success=success,
            **{k: v for k, v in kwargs.items() if k in ("blueprints", "agents", "agent")},
        )

    async def aprocess(
        self,
        messages: list[Any],
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        team_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Async version of process(). See process() docstring for kwargs."""
        self._template_stored = False

        success         = kwargs.get("success", True)
        execution_trace = kwargs.get("execution_trace")
        keyword         = kwargs.get("keyword")

        if keyword:
            await self.config.plan_cache.arecord_outcome(keyword, success)

        if not success:
            return

        if execution_trace:
            await self.aextract_and_cache(
                messages=messages,
                execution_trace=execution_trace,
                session_id=session_id,
                user_id=user_id,
                agent_id=agent_id,
                team_id=team_id,
                keyword=keyword,
                tool_chain=kwargs.get("tool_chain"),
                task_description=kwargs.get("task_description"),
                aliases=kwargs.get("aliases"),
            )

        self._link_blueprints_to_template(
            messages=messages,
            keyword=keyword,
            success=success,
            **{k: v for k, v in kwargs.items()
               if k in ("blueprints", "agents", "agent")},
        )

    def extract_and_cache(
        self,
        messages: list["Message"],
        execution_trace: list,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        team_id: str | None = None,
        keyword: str | None = None,
        tool_chain: list[str]  | None = None,
        task_description: str | None= None,
        aliases: list[str]  | None  = None,
        **kwargs: Any,
    ) -> str:
        log_debug("PlanCacheStore: extracting plan template", center=True)

        if not keyword:
            query   = self._messages_to_query(messages)
            keyword = self._extract_keyword(query) if query else None

        if not keyword:
            log_warning("PlanCacheStore.extract_and_cache: could not derive keyword.")
            return "Could not extract keyword"

        summary_prompt = dedent(f"""\
            Could you please summary the whole plan within around 25-50 words? The summary should be succint, and concise.
            
            Task description: 
            {task_description}
            
            Plan:
            {execution_trace}
            
            Tool use: 
            {tool_chain}
        """)
        model_copy = deepcopy(self.model)
        response = model_copy.response( #type:ignore
            messages=[Message(role="user", content=summary_prompt)]  
        )
        summary = response.content.strip().lower() if response.content else "No summary available."

        template = build_plan_template_from_trace(
            keyword=keyword,
            task_description=task_description or f"Task: {keyword}",
            worker_results=execution_trace,
            tool_chain=tool_chain or [],
            aliases=aliases,
            summary=summary,
        )
        self.config.plan_cache.store_template(template)
        self._template_stored = True
        log_debug(f"PlanCacheStore: cached template '{keyword}' ({len(template.steps)} steps)")
        return f"Plan template cached: {keyword}"

    async def aextract_and_cache(
        self,
        messages: list["Message"],
        execution_trace: list,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        team_id: str | None = None,
        keyword: str | None = None,
        tool_chain: list[str]  | None = None,
        task_description: str | None= None,
        aliases: list[str]  | None  = None,
        **kwargs: Any,
    ) -> str:
        log_debug("PlanCacheStore: extracting plan template (async)", center=True)

        if not keyword:
            query   = self._messages_to_query(messages)
            keyword = await self._aextract_keyword(query) if query else None

        if not keyword:
            log_warning("PlanCacheStore.aextract_and_cache: could not derive keyword.")
            return "Could not extract keyword"

        summary_prompt = dedent(f"""\
            Could you please summary the whole plan within around 25-50 words? The summary should be succint, and concise.
            
            Task description: 
            {task_description}
            
            Plan:
            {execution_trace}
            
            Tool use: 
            {tool_chain}
        """)
        model_copy = deepcopy(self.model)
        response = await model_copy.aresponse(  #type:ignore
            messages=[Message(role="user", content=summary_prompt)]  
        )
        summary = response.content.strip().lower() if response.content else "No summary available."
        
        template = build_plan_template_from_trace(
            keyword=keyword,
            task_description=task_description or f"Task: {keyword}",
            worker_results=execution_trace,
            tool_chain=tool_chain or [],
            aliases=aliases,
            summary=summary,
        )

        await self.config.plan_cache.astore_template(template)
        self._template_stored = True
        log_debug(f"PlanCacheStore: cached template '{keyword}' ({len(template.steps)} steps)")
        return f"Plan template cached: {keyword}"

    def get_tools(
        self,
        data: Optional[CacheLookupResult] = None,
        **kwargs: Any,
    ) -> list[Callable]:
        """
        Return adaptation tools for the agent.

        ``data`` is the CacheLookupResult returned by recall().  When called
        by agno's LearningMachine, data arrives via the positional ``data``
        kwarg that the machine passes after recall.

        All tool closures capture a *snapshot* of data (not instance state)
        so they remain correct even if another request triggers a new recall.
        """
        if not self.config.enable_adaptation_tools:
            return []

        tools: list[Callable] = []

        if data and data.hit and data.template:
            tools.append(self._create_adapt_plan_tool(data.template))

        tools.append(self._create_search_plans_tool())

        if data and len(data.candidates) > 1:
            tools.append(self._create_view_alternatives_tool(data.candidates))

        return tools

    async def aget_tools(
        self,
        data: Optional[CacheLookupResult] = None,
        **kwargs: Any,
    ) -> list[Callable]:
        return self.get_tools(data=data, **kwargs)

    # ------------------------------------------------------------------
    # Tool factories  (accept explicit arguments instead of reading state)
    # ------------------------------------------------------------------

    def _create_adapt_plan_tool(self, template: PlanTemplate) -> Callable:
        """Closure captures a snapshot of the template, not instance state."""

        def adapt_plan_template(
            entity_mappings: dict[str, str],
            step_customizations: Optional[dict[int, str]] = None,
        ) -> str:
            """
            Adapt the cached plan template to the current task.

            Args:
                entity_mappings: dict mapping placeholder names (e.g. "[Company]")
                    to their actual values for this task.
                step_customizations: Optional dict mapping step indices to
                    extra instructions to append for that step.

            Returns:
                Formatted, adapted plan as a numbered list.
            """
            adapted: list[str] = []
            for step in template.get_steps():
                instruction = step.instruction
                for placeholder, value in entity_mappings.items():
                    instruction = instruction.replace(placeholder, value)
                if step_customizations and step.step_index in step_customizations:
                    instruction += f" ({step_customizations[step.step_index]})"
                adapted.append(f"{step.step_index + 1}. {instruction}")
            return "\n".join(adapted)

        return adapt_plan_template

    def _create_search_plans_tool(self) -> Callable:
        cache = self.config.plan_cache

        def search_similar_plans(
            query: str,
            mode: Optional[str] = None,
            limit: int = 5,
        ) -> str:
            """
            Search the plan cache for templates similar to *query*.

            Args:
                query: Natural-language description of the task.
                mode:  Retrieval mode override ("exact", "fuzzy",
                       "embedding", "hybrid").  Uses the cache default if None.
                limit: Maximum number of results to display.

            Returns:
                Formatted list of matching plan templates.
            """
            retrieval_mode = RetrievalMode(mode) if mode else None
            result = cache.lookup(
                query=query,
                limit=limit,
                mode=retrieval_mode,
            )

            if not result.hit and not result.candidates:
                return f"No plans found for: {query}"

            hits = (
                [(result.template, result.confidence)] if result.hit and result.template else []
            ) + result.candidates

            lines = [f"Found {len(hits)} matching plan(s):\n"]
            for tmpl, score in hits[:limit]:
                lines.append(
                    f"- {tmpl.keyword}: {score:.0%} match, "
                    f"{tmpl.hit_rate():.0%} success rate, "
                    f"{len(tmpl.steps)} steps"
                )
            return "\n".join(lines)

        return search_similar_plans

    def _create_recent_plans(self) -> Callable:
        cache = self.config.plan_cache
        
        def list_recent_plans(limit: int = 5) -> str:
            """
            Retrieve a list of the most recently cached and successful plan templates.
            Useful for identifying current trends in task execution.
            """
            templates = cache.get_all_templates()
            recent = sorted(
                templates, key=lambda t: t.last_used_at or 0, reverse=True
            )
            if not recent:
                return "No recently used plans found."
            
            lines = [f"Recently used plans:\n"]
            for tmpl in recent[:limit]:
                last_used = tmpl.last_used_at or 0
                lines.append(
                    f"- {tmpl.keyword} (last used: {last_used}, "
                    f"success rate: {tmpl.hit_rate():.0%}, "
                    f"steps: {len(tmpl.steps)})"
                )
            return "\n".join(lines)
        return list_recent_plans

    def _create_top_plans_tool(self) -> Callable:
        cache = self.config.plan_cache
        
        def list_top_plans(limit: int = 5) -> str:
            """
            Retrieve a list of the top-performing plans by success rate.
            """
            templates = cache.get_all_templates()
            top = sorted(
                templates, key=lambda t: t.hit_rate(), reverse=True
            )
            if not top:
                return "No plans found."
            
            lines = [f"Top-performing plans:\n"]
            for tmpl in top[:limit]:
                lines.append(
                    f"- {tmpl.keyword} (success rate: {tmpl.hit_rate():.0%}, "
                    f"used: {tmpl.total_uses()}, "
                    f"steps: {len(tmpl.steps)})"
                )
            return "\n".join(lines)
        return list_top_plans
    
    def _create_inspect_plan_tool(self) -> Callable:
        cache = self.config.plan_cache
        
        def inspect_plan(keyword: str) -> str:
            """
            Retrieve detailed information about a specific plan template.

            Args:
                keyword: The keyword of the plan to inspect.

            Returns:
                Detailed information about the plan template.
            """
            template = cache.get_template(keyword)
            if not template:
                return f"No plan found with keyword: {keyword}"
            
            lines = [
                f"Plan Keyword: {template.keyword}",
                f"Task Description: {template.task_description}",
                f"Success Rate: {template.hit_rate():.0%} ({template.successes}/{template.total_uses()} uses)",
                f"Steps ({len(template.steps)}):"
            ]
            for step in template.get_steps():
                lines.append(f"  {step.step_index + 1}. {step.instruction}")
            return "\n".join(lines)
        
        return inspect_plan
    
    def _create_view_alternatives_tool(
        self,
        candidates: list[tuple],
    ) -> Callable:
        """Closure captures a snapshot of candidates, not instance state."""
        snapshot = list(candidates)

        def view_alternative_plans(limit: int = 5) -> str:
            """
            list alternative plan templates that were close matches in the last
            cache lookup.

            Args:
                limit: Maximum number of alternatives to display.

            Returns:
                Formatted list of alternative templates.
            """
            if not snapshot:
                return "No alternative plans available."

            lines = ["Alternative plan templates:\n"]
            for i, (tmpl, score) in enumerate(snapshot[:limit], 1):
                snippet = (tmpl.task_description or "")[:100]
                lines.append(
                    f"{i}. {tmpl.keyword} (score: {score:.0%})\n"
                    f"   Success rate: {tmpl.hit_rate():.0%}, "
                    f"Steps: {len(tmpl.steps)}\n"
                    f"   {snippet}{'…' if len(tmpl.task_description) > 100 else ''}"
                )
            return "\n".join(lines)

        return view_alternative_plans

    # ------------------------------------------------------------------
    # Keyword extraction  (sync + async)
    # ------------------------------------------------------------------

    def _extract_keyword(self, query: str) -> Optional[str]:
        """
        Extract a high-level task keyword using the configured model.
        Falls back to the raw query if no model is configured.
        """
        if not self.model:
            log_debug("PlanCacheStore: no model configured, using raw query as keyword.")
            return query.strip().lower()[:80]

        try:
            extraction_prompt = dedent(f"""\
                Extract a high-level task-type keyword from the query below.
                Focus on WHAT (the task type), not WHO / WHEN / WHERE (specifics).

                Examples:
                  "Calculate working capital ratio for Tesla Q4 2024"
                    → "working capital ratio"
                  "Find Python tutorial videos from Corey Schafer"
                    → "search videos by topic"
                  "Analyse Apple's latest earnings report"
                    → "earnings analysis"

                Query: {query}

                Respond with ONLY the keyword phrase (2–6 words, lowercase).
            """)
            model_copy = deepcopy(self.model)
            response   = model_copy.response(
                messages=[Message(role="user", content=extraction_prompt)]  # type: ignore[arg-type]
            )
            keyword = response.content.strip().lower() if response.content else None
            log_debug(f"PlanCacheStore: extracted keyword='{keyword}'")
            return keyword
        except Exception as e:
            log_warning(f"PlanCacheStore._extract_keyword failed: {e}")
            return query.strip().lower()[:80]

    async def _aextract_keyword(self, query: str) -> Optional[str]:
        """Async keyword extraction."""
        if not self.model:
            log_debug("PlanCacheStore: no model configured, using raw query as keyword.")
            return query.strip().lower()[:80]

        try:
            extraction_prompt = dedent(f"""\
                Extract a high-level task-type keyword from the query below.
                Focus on WHAT (the task type), not WHO / WHEN / WHERE (specifics).

                Examples:
                  "Calculate working capital ratio for Tesla Q4 2024"
                    → "working capital ratio"
                  "Find Python tutorial videos from Corey Schafer"
                    → "search videos by topic"
                  "Analyse Apple's latest earnings report"
                    → "earnings analysis"

                Query: {query}

                Respond with ONLY the keyword phrase (2–6 words, lowercase).
            """)
            model_copy = deepcopy(self.model)
            response   = await model_copy.aresponse(
                messages=[Message(role="user", content=extraction_prompt)]  # type: ignore[arg-type]
            )
            keyword = response.content.strip().lower() if response.content else None
            log_debug(f"PlanCacheStore: extracted keyword='{keyword}'")
            return keyword
        except Exception as e:
            log_warning(f"PlanCacheStore._aextract_keyword failed: {e}")
            return query.strip().lower()[:80]

    # ------------------------------------------------------------------
    # Blueprint linking helper
    # ------------------------------------------------------------------

    def _link_blueprints_to_template(
        self,
        messages: list[Any],
        keyword:  Optional[str] = None,
        success:  bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Delegate blueprint storage to blueprint_store and write the resulting
        blueprint_ids back into the matched PlanTemplate.

        Accepts:
          blueprints : list[AgentBlueprint]  – pre-built blueprints
          agents     : list[Agent]           – live agents to extract specs from
          agent      : Agent                 – single live agent
        """
        if self.blueprint_store is None or not keyword:
            return

        blueprints = list(kwargs.get("blueprints") or [])
        agents     = list(kwargs.get("agents") or [])
        if kwargs.get("agent"):
            agents.append(kwargs["agent"])

        new_ids: list[str] = []

        for bp in blueprints:
            self.blueprint_store.config.blueprint_cache.store_blueprint(bp)
            new_ids.append(bp.blueprint_id)

        for ag in agents:
            try:
                from agent_blueprint_cache.components import build_blueprint_from_agent as _bba
                bp = _bba(
                    agent=ag,
                    task_keyword=keyword,
                    generated_for_query=self._messages_to_query(messages),
                )
                self.blueprint_store.config.blueprint_cache.store_blueprint(bp)
                new_ids.append(bp.blueprint_id)
            except Exception as e:
                log_warning(f"PlanCacheStore._link_blueprints: could not extract blueprint: {e}")

        if not new_ids:
            return

        template = self.config.plan_cache.get_template(keyword)
        if template:
            existing = set(template.blueprint_ids)
            template.blueprint_ids = list(existing | set(new_ids))
            self.config.plan_cache.store_template(template)
            log_debug(
                f"PlanCacheStore: linked {len(new_ids)} blueprint(s) "
                f"to plan '{keyword}'"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _messages_to_query(self, messages: list["Message"]) -> str:
        """Extract the most recent user message as the query string."""
        for msg in reversed(messages):
            if msg.role == "user":
                content = (
                    msg.get_content_string()
                    if hasattr(msg, "get_content_string")
                    else str(msg.content or "")
                )
                if content.strip():
                    return content.strip()
        return ""

    def set_log_level(self) -> None:
        if self.debug_mode or getenv("AGNO_DEBUG", "false").lower() == "true":
            self.debug_mode = True
            set_log_level_to_debug()
        else:
            set_log_level_to_info()

    # ------------------------------------------------------------------
    # Debug / diagnostics
    # ------------------------------------------------------------------

    def print_cache_stats(self) -> None:
        """Pretty-print cache statistics to stdout."""
        from agno.learn.utils import print_panel  # type: ignore[attr-defined]

        cache = self.config.plan_cache
        if not cache:
            print("No plan cache configured.")
            return

        stats = cache.get_statistics()
        lines = [
            f"Retrieval Mode  : {stats['retrieval_mode']}",
            f"Qdrant Size     : {stats['cache_size']}",
            f"In-Memory Size  : {stats['in_memory_size']}",
            f"Cache Hits      : {stats['cache_hits']}",
            f"Cache Misses    : {stats['cache_misses']}",
            f"Hit Rate        : {stats['hit_rate']}",
            f"Evictions       : {stats['evictions']}",
            f"Avg Success Rate: {stats['average_success_rate']}",
            "",
            "Top Templates:",
        ]
        for template in cache.get_all_templates()[:10]:
            lines.append(
                f"  - {template.keyword}: {template.hit_rate():.0%} "
                f"({template.total_uses()} uses)"
            )

        try:
            print_panel(title="Plan Cache Statistics", subtitle="", lines=lines, raw=False)
        except TypeError:
            print("\n".join(lines))

    def __repr__(self) -> str:
        cache_size = self.config.plan_cache.size if self.config.plan_cache else 0
        mode       = self.config.retrieval_mode.value
        return (
            f"PlanCacheStore("
            f"mode={mode}, "
            f"cached_plans={cache_size})"
        )