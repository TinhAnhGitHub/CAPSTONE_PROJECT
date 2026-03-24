from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from agno.tools import Toolkit


@dataclass
class ToolDocumentation:
    """Documentation for a single tool."""
    name: str
    toolkit_name: str
    description: str
    instructions: str
    parameters: dict[str, Any]
    returns: str
    usage_examples: list[str] = field(default_factory=list)


class ToolRegistry:
    """Registry that collects and documents all available tools.

    This registry introspects all toolkits and extracts tool metadata
    (descriptions, parameters, instructions) to provide as documentation
    to planning agents.

    Usage:
        registry = ToolRegistry()
        registry.register_toolkit(video_search_toolkit)
        registry.register_toolkit(utility_toolkit)

        # Get documentation for planning agent
        docs = registry.generate_planning_context()
    """

    def __init__(self):
        self._toolkits: dict[str, Toolkit] = {}
        self._tools: dict[str, ToolDocumentation] = {}

    def register_toolkit(self, toolkit: Toolkit, alias: str | None = None) -> None:
        """Register a toolkit and extract all its tools.

        Args:
            toolkit: The Toolkit instance to register
            alias: Optional alias for the toolkit (defaults to class name)
        """
        toolkit_name = alias or toolkit.__class__.__name__
        self._toolkits[toolkit_name] = toolkit
        self._extract_tools_from_toolkit(toolkit=toolkit, toolkit_name)

    def _extract_tools_from_toolkit(self, toolkit: Toolkit, toolkit_name: str) -> None:
        """Extract tool documentation from a toolkit instance.

        Uses agno's internal functions dict for registered tools.
        """
        if hasattr(toolkit, 'functions') and toolkit.functions:
            for func_name, func in toolkit.functions.items():
                self._register_function(func, toolkit_name, func_name)
        else:
            self._extract_via_introspection(toolkit, toolkit_name)

    def _register_function(self, func: Any, toolkit_name: str, func_name: str) -> None:
        """Register a tool function from agno's internal registry."""
        description = getattr(func, 'description', '') or ''
        instructions = getattr(func, 'instructions', '') or ''
        parameters = self._extract_parameters_from_func(func)

        tool_doc = ToolDocumentation(
            name=func_name,
            toolkit_name=toolkit_name,
            description=description,
            instructions=instructions,
            parameters=parameters,
            returns=self._extract_return_doc(func),
        )
        self._tools[f"{toolkit_name}.{func_name}"] = tool_doc

    def _extract_parameters_from_func(self, func: Any) -> dict[str, Any]:
        """Extract parameter info from agno Function object."""
        parameters = {}
        if hasattr(func, 'parameters_schema') and func.parameters_schema:
            schema = func.parameters_schema
            properties = schema.get('properties', {})
            required = schema.get('required', [])
            for param_name, param_schema in properties.items():
                parameters[param_name] = {
                    'type': param_schema.get('type', 'any'),
                    'description': param_schema.get('description', ''),
                    'required': param_name in required,
                    'default': param_schema.get('default'),
                }
        return parameters

    def _extract_via_introspection(self, toolkit: Toolkit, toolkit_name: str) -> None:
        """Fallback: extract tools via Python introspection."""
        for name, method in inspect.getmembers(toolkit, predicate=inspect.ismethod):
            if hasattr(method, '_is_tool') or hasattr(method, 'description'):
                self._register_method(method, toolkit_name, name)

    def _register_method(self, method: Any, toolkit_name: str, method_name: str) -> None:
        """Register a tool method via introspection."""
        sig = inspect.signature(method)
        docstring = inspect.getdoc(method) or ""
        description = getattr(method, 'description', '')
        instructions = getattr(method, 'instructions', '')

        parameters = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            parameters[param_name] = {
                'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any',
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'required': param.default == inspect.Parameter.empty,
            }

        tool_doc = ToolDocumentation(
            name=method_name,
            toolkit_name=toolkit_name,
            description=description,
            instructions=instructions,
            parameters=parameters,
            returns=self._parse_returns(docstring),
        )
        self._tools[f"{toolkit_name}.{method_name}"] = tool_doc

    def _extract_return_doc(self, func: Any) -> str:
        """Extract return documentation from function."""
        if hasattr(func, 'returns') and func.returns:
            return func.returns
        return "ToolResult with content"

    def _parse_returns(self, docstring: str) -> str:
        """Parse Returns section from docstring."""
        if 'Returns:' in docstring:
            return docstring.split('Returns:')[-1].split('\n\n')[0].strip()
        return "ToolResult with content"

    def get_tool_documentation(self, tool_name: str) -> ToolDocumentation | None:
        """Get documentation for a specific tool."""
        return self._tools.get(tool_name)

    def get_all_tools(self) -> list[ToolDocumentation]:
        """Get all registered tool documentation."""
        return list(self._tools.values())

    def get_tools_by_toolkit(self, toolkit_name: str) -> list[ToolDocumentation]:
        """Get all tools for a specific toolkit."""
        return [t for t in self._tools.values() if t.toolkit_name == toolkit_name]

    def list_tool_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def generate_documentation(self, format: str = "markdown") -> str:
        """Generate comprehensive documentation for all tools."""
        if format == "markdown":
            return self._generate_markdown_docs()
        elif format == "json":
            import json
            return json.dumps([t.__dict__ for t in self._tools.values()], indent=2)
        return self._generate_text_docs()

    def _generate_markdown_docs(self) -> str:
        """Generate markdown documentation."""
        lines = ["# Available Tools Documentation\n"]
        by_toolkit: dict[str, list[ToolDocumentation]] = {}
        for tool in self._tools.values():
            by_toolkit.setdefault(tool.toolkit_name, []).append(tool)

        for toolkit_name, tools in sorted(by_toolkit.items()):
            lines.append(f"\n## {toolkit_name}\n")
            for tool in sorted(tools, key=lambda t: t.name):
                lines.append(f"### `{tool.name}`\n")
                lines.append(f"**Description:** {tool.description}\n")
                if tool.instructions:
                    lines.append(f"**When to use:** {tool.instructions}\n")
                if tool.parameters:
                    lines.append("**Parameters:**\n")
                    for param_name, param_info in tool.parameters.items():
                        required = "*" if param_info.get('required') else ""
                        lines.append(f"- `{param_name}`{required}: {param_info.get('type', 'Any')}")
                        if param_info.get('default') is not None:
                            lines[-1] += f" (default: {param_info['default']})"
                lines.append(f"\n**Returns:** {tool.returns}\n")
                lines.append("---\n")
        return "\n".join(lines)

    def _generate_text_docs(self) -> str:
        """Generate plain text documentation for LLM context."""
        lines = ["AVAILABLE TOOLS\n===============\n"]
        for tool in self._tools.values():
            lines.append(f"\n{tool.toolkit_name}.{tool.name}")
            lines.append(f"  Description: {tool.description}")
            if tool.instructions:
                lines.append(f"  When to use: {tool.instructions}")
            if tool.parameters:
                lines.append("  Parameters:")
                for param_name, param_info in tool.parameters.items():
                    req = " (required)" if param_info.get('required') else ""
                    lines.append(f"    - {param_name}: {param_info.get('type', 'Any')}{req}")
            lines.append(f"  Returns: {tool.returns}")
        return "\n".join(lines)

    def generate_planning_context(self) -> str:
        """Generate optimized context for planning agents."""
        lines = [
            "## Available Tools\n",
            "You have access to the following tool categories:\n",
        ]
        by_toolkit: dict[str, list[ToolDocumentation]] = {}
        for tool in self._tools.values():
            by_toolkit.setdefault(tool.toolkit_name, []).append(tool)

        for toolkit_name, tools in sorted(by_toolkit.items()):
            lines.append(f"### {toolkit_name}")
            for tool in sorted(tools, key=lambda t: t.name):
                short_desc = tool.description.split('.')[0] if tool.description else "No description"
                lines.append(f"- **{tool.name}**: {short_desc}")
            lines.append("")
        lines.append("\nUse these tools to execute your plan. Each tool returns a ToolResult with content.")
        return "\n".join(lines)


_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _registry
    _registry = None
