from typing import Callable, Dict, List, Optional, Any, get_type_hints
from functools import wraps, partial
import inspect
from llama_index.core.tools import FunctionTool

class ToolMetadata:
    """Metadata extracted from tool function"""
    def __init__(
        self,
        func: Callable,
        category: str,
        tags: List[str],
        dependencies: List[str]
    ):
        self.func = func
        self.category = category
        self.tags = tags
        self.dependencies = dependencies
        self.name = func.__name__
        self.signature = inspect.signature(func)
        self.docstring = inspect.getdoc(func)
        self.type_hints = get_type_hints(func)
    
    def get_user_params(self) -> Dict[str, inspect.Parameter]:
        """Get parameters that are NOT dependencies (what LLM controls)"""
        params = {}
        for name, param in self.signature.parameters.items():
            if name not in self.dependencies:
                params[name] = param
        return params
    
    def to_dict(self) -> dict:
        """Serialize for documentation/inspection"""
        return {
            'name': self.name,
            'category': self.category,
            'tags': self.tags,
            'dependencies': self.dependencies,
            'docstring': self.docstring,
            'user_params': {
                name: {
                    'type': str(param.annotation),
                    'default': param.default if param.default != inspect.Parameter.empty else None
                }
                for name, param in self.get_user_params().items()
            }
        }
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
        self._tools = {}
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
        """List all tool names"""
        return list(self._tools.keys())

    def list_all_tags(self) -> list[str]:
        return list(self._tags)

    def list_all_categories(self) -> list[str]:
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
    
    def generate_docs(self, category: Optional[str] = None) -> str:
        """Generate markdown documentation"""
        lines = ["# Tool Registry Documentation\n"]
        
        categories = [category] if category else sorted(self._categories.keys())
        
        for cat in categories:
            lines.append(f"\n## {cat.title()}\n")
            
            for tool_name in self._categories.get(cat, []):
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
        
        return "".join(lines)

tool_registry = ToolRegistry()