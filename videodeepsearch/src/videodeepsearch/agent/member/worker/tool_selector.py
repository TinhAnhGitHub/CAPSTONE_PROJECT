# agents/member/worker/tool_selector.py
#
# ToolSelector sits between the toolkit registry and the spawned workers.
# Instead of giving every worker every tool, the Orchestrator (or Planner)
# specifies EXACTLY which tool names each worker needs.
#
# Agno's Agent(tools=[...]) accepts:
#   - Toolkit  (all tools in it)
#   - Function (a single extracted tool)
#   - Callable (raw Python function)
#   - Dict     (raw OpenAI-style schema)

from loguru import logger
from dataclasses import dataclass, field
from collections.abc import Callable
from collections import defaultdict
 
from agno.tools.function import Function
from agno.tools.toolkit import Toolkit



@dataclass
class ToolSelector:
    """
    Registry of toolkits that can produce per-worker tool subsets.
 
    Usage
    -----
    selector = ToolSelector()
    selector.register("search", search_toolkit_factory)
    selector.register("utility", utility_toolkit_factory)
 
    # At worker spawn time:
    tools = selector.resolve([
        "search.get_images_from_qwenvl_query",
        "search.get_segments_from_event_query_mmbert",
        "utility.store_evidence",
    ])
    worker = Agent(..., tools=tools)
 
    Notes
    -----
    - Factories are called lazily (once per worker spawn) so each worker
      gets a fresh toolkit instance with its own isolated state.
    - If a tool name is not found, it is logged and skipped (never raises)
      so a bad tool name in the Orchestrator's plan doesn't crash the run.
    - Toolkit instances created by the factory are NOT cached — each
      resolve() call produces fresh instances. This matches the isolation
      requirement of VideoSearchToolkit._result_store.
    """
    
    
    _factories: dict[str, Callable[[], Toolkit]] = field(default_factory=dict)
    
    def register(
        self,
        name: str,
        factory: Callable[[], Toolkit]
    ):
        self._factories[name] = factory
        logger.debug(f"[ToolSelector] Registered toolkit factory: {name!r}")
    
    def resolve(
        self,
        tool_names: list[str]
    )-> list[Function]:
        """
        Resolve a list of "toolkit_alias.tool_name" strings into Function objects.
 
        Args:
            tool_names: List of strings in the format "alias.tool_name".
                        Example: ["search.get_images_from_qwenvl_query",
                                  "search.view_cache_result_from_handle_id"]
 
        Returns:
            List of agno Function objects ready to pass to Agent(tools=[...]).
            Unknown names are skipped with a warning.
        """
        by_toolkit = defaultdict(list[str])
        
        for entry in tool_names:
            if  '.' not in entry:
                logger.warning(
                    f"[ToolSelector] Tool name {entry!r} has no alias prefix "
                    f"(expected 'alias.tool_name'). Skipping."
                )
                continue
                
            alias, func_name = entry.split('.')
            
            by_toolkit[alias].append(func_name)
        
        resolved: list[Function] = []
        for alias, names in by_toolkit.items():
            factory = self._factories.get(alias)
            if factory is None:
                logger.warning(
                    f"[ToolSelector] No toolkit registered under alias {alias!r}. "
                    f"Skipping tools: {names}"
                )
                continue
        
            toolkit_instance = factory()
            all_functions: dict[str, Function] = {
                **toolkit_instance.functions,
                **toolkit_instance.async_functions,
            }
            
            for func_name in names:
                func = all_functions.get(func_name)
                if func is None:
                    available = sorted(all_functions.keys())
                    logger.warning(
                        f"[ToolSelector] Tool {func_name!r} not found in "
                        f"toolkit {alias!r}. Available: {available}"
                    )
                    continue
                resolved.append(func)
                logger.debug(f"[ToolSelector] Resolved: {alias}.{func_name}")
            
        
        return resolved
    
    def list_all(self) -> dict[str, list[str]]:
        """
        Return all tool names grouped by toolkit alias.
        Useful for the Planner to know what is available.
 
        Note: calls each factory once — use sparingly.
        """
        result: dict[str, list[str]] = {}
        for alias, factory in self._factories.items():
            try:
                instance = factory()
                all_funcs = {**instance.functions, **instance.async_functions}
                result[alias] = sorted(all_funcs.keys())
            except Exception as e:
                logger.error(f"[ToolSelector] list_all failed for {alias!r}: {e}")
                result[alias] = [f"<error: {e}>"]
        return result
 