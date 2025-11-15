from typing import Callable, Dict, List, Optional, Any, get_type_hints, get_origin, get_args, Annotated
from collections.abc import Sequence as SequenceABC
from pydantic import BaseModel
import inspect
from llama_index.core.tools import FunctionTool
from dataclasses import dataclass

from .helper import _format_model_doc



@dataclass(frozen=True)
class ToolMetadata:
    """Metadata extracted from tool function"""
    func: Callable
    category: str
    tags: List[str]
    dependencies: List[str]

    @property
    def name(self) -> str:
        """Function name"""
        return self.func.__name__
    
    @property
    def signature(self) -> inspect.Signature:
        """Function signature"""
        return inspect.signature(self.func)

    @property
    def docstring(self) -> str | None:
        """Function docstring"""
        return inspect.getdoc(self.func)
    
    @property
    def type_hints(self) -> dict[str, Any]:
        """Function type hints"""
        return get_type_hints(self.func)
    
    @property
    def output_type(self) -> Optional[Any]:
        return self.type_hints.get("return")
    
    @property
    def return_fields_doc(self) -> str:
        ret_type = self.output_type
        if ret_type is None:
            return "This function does not have a return Type, might be python void function"

        origin = get_origin(ret_type)
        args = get_args(ret_type)

        if origin is Annotated:
            ret_type = args[0]
            origin = get_origin(ret_type)
            args = get_args(ret_type)

        origin_is_sequence = False
        if origin is not None:
            try:
                origin_is_sequence = issubclass(origin, SequenceABC)
            except TypeError:
                origin_is_sequence = False

        if origin_is_sequence and origin is not tuple:
            inner_type = args[0] if args else Any
            if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel): #type:ignore
                return _format_model_doc(inner_type, container='list')
            else:
                tname = getattr(inner_type, "__name__", str(inner_type))
                return f"Returns a list of `{tname}` values."
        
        # --- tuple[types...] ---
        if origin is tuple:
            inner_types = ", ".join(getattr(a, "__name__", str(a)) for a in args) or Any
            return f"Returns a tuple({inner_types})."
        
        # --- BaseModel ---
        if inspect.isclass(ret_type) and issubclass(ret_type, BaseModel):
            return _format_model_doc(ret_type)
        return f"Returns a `{getattr(ret_type, '__name__', str(ret_type))}` value."
    
    def get_user_params(self) -> Dict[str, inspect.Parameter]:
        """Get parameters that are NOT dependencies (what LLM controls)"""
        params = {}
        for name, param in self.signature.parameters.items():
            if name not in self.dependencies:
                params[name] = param
        return params

    
class ToolRegistry:
    """
    Central registry with decorator-based registration
    Separates tool discovery from dependency injection
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
            cls._instance._categories = {}
        return cls._instance
    
    def __init__(self):
        self._tools: dict[str, ToolMetadata] = {}
        self._categories = {}
        self._tags = set()
        
    def register(
        self,
        category: str,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None
    ):
        """
        Decorator to register a tool
        
        Example:
            @tool_registry.register(
                category="search",
                tags=["visual", "semantic"],
                dependencies=["milvus_client", "user_id"]
            )
            async def my_search_tool(query: str, milvus_client, user_id):
                pass
        """
        def decorator(func: Callable) -> Callable:
            metadata = ToolMetadata(
                func=func,
                category=category,
                tags=tags or [],
                dependencies=dependencies or []
            )
            if tags:
                self._tags.update(tags)
            
            self._tools[func.__name__] = metadata
            
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(func.__name__)
            
            return func  # Return unchanged
        
        return decorator
    
    def get(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata by name"""
        return self._tools.get(name)
    
    def list_all(self) -> List[str]:
        """
        Use this tools to list all the tools available in the system
        """
        return list(self._tools.keys())

    def list_all_tags(self) -> list[str]:
        """
        Use this tool to list all the available tags that associate with the predefined tools.
        """
        return list(self._tags)

    def list_all_categories(self) -> list[str]:
        """List all categories that contain registered tools."""
        return list(self._categories.keys())

    def list_by_category(self, category: str) -> List[str]:
        """List tools in a category"""
        return self._categories.get(category, [])
    
    def list_by_tags(self, tags: List[str]) -> List[str]:
        """List tools with any of the given tags"""
        result = []
        for name, metadata in self._tools.items():
            if any(tag in metadata.tags for tag in tags):
                result.append(name)
        return result
    
    def _generate_doc(self, tool_name:str) -> list[str]:
        lines = []
        metadata = self._tools[tool_name]
        lines.append(f"\n### `{tool_name}`\n")

        if metadata.docstring:
            lines.append(f"{metadata.docstring}\n")
        lines.append("\n**User Parameters:**\n")
        for param_name, param in metadata.get_user_params().items():
            param_type = param.annotation
            default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
            lines.append(f"- `{param_name}: {param_type}{default}`\n")
        
        lines.append(f"\n**Required Dependencies:** {', '.join(metadata.dependencies)}\n")
        lines.append(f"**Tags:** {', '.join(metadata.tags)}\n")
        lines.append(f"Output return: {metadata.return_fields_doc}\n\n")
        lines.append("\n" + "#"*10 + "\n")
        return lines

    def generate_docs_all_functions(self) -> str:
        """
        IMPORTANT: MUST USE THIS TOOL. Generating all the documentation related to the system tools' capability. 
        """
        lines = ["# Tool Registry Documentation\n"]
        
        for tool_name in self.list_all():
            tool_doc = self._generate_doc(tool_name)
            lines.extend(tool_doc)
        return "".join(lines)
        
    def generate_docs(self, category: Optional[str] = None) -> str:
        """Generate markdown documentation for all registered tools."""
        lines = ["# Tool Registry Documentation\n"]
        categories = [category] if category else sorted(self._categories.keys())
        for cat in categories:
            lines.append(f"\n## {cat.title()}\n")
            
            for tool_name in self._categories.get(cat, []):
                tool_doc = self._generate_doc(tool_name)
                lines.extend(tool_doc)
        return "".join(lines)

  

tool_registry = ToolRegistry()


def get_registry_tools() -> list[FunctionTool]:
    """
    Return all FunctionTool objects exposing the ToolRegistry interface
    for ReAct or other agent planning systems.
    """
    return [
        FunctionTool.from_defaults(fn=tool_registry.list_all),
        FunctionTool.from_defaults(fn=tool_registry.list_all_tags),
        FunctionTool.from_defaults(fn=tool_registry.list_all_categories),
        FunctionTool.from_defaults(fn=tool_registry.list_by_category),
        FunctionTool.from_defaults(fn=tool_registry.list_by_tags),
        FunctionTool.from_defaults(fn=tool_registry.get),
        FunctionTool.from_defaults(fn=tool_registry.generate_docs_all_functions),
        FunctionTool.from_defaults(fn=tool_registry.generate_docs),
    ]
