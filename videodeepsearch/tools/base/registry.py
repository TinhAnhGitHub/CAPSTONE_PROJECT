from __future__ import annotations
from typing import Callable, cast
from pydantic import BaseModel, Field
import inspect
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import ToolMetadata, FunctionTool

from .doc_template.group_doc import GroupName, GROUPNAME_TO_TEMPLATE
from .types import ExtentedToolMetadata, BundleSpec, BundleRoles

from videodeepsearch.agent.base import get_global_agent_registry

class GroupDoc:
    def __init__(self, name: str, group_doc_template: RichPromptTemplate):
        self.name = name
        self.group_doc_template = group_doc_template
        self._related_tools: list[ExtentedToolMetadata] = []
    
    def add_tool_metadata(self, tool_metadata: ExtentedToolMetadata):
        self._related_tools.append(tool_metadata)

    def generate_document(self) -> str:
        tool_info_parts = []

        for extended_tool_metadata in self._related_tools:

            
            tool_name = extended_tool_metadata.tool_name
            tool_repr = f"""
            - **{tool_name}**
            """
            tool_info_parts.append(tool_repr)

        tool_info = "\n".join(tool_info_parts)

        doc_template = self.group_doc_template.format(tool_usage=tool_info)
        return doc_template
    
    def is_tool_belong(self, tool_name) -> bool:

        return any(
            tool_metadata.tool_name == tool_name for tool_metadata in self._related_tools #type:ignore
        )

class ToolRegistry:
    """
    Central Tool Registry with decorator-based registration
    Separates tool discovery
    """
    _instance = None
    

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


    def __init__(self):
        self.group_docs: dict[str, GroupDoc] = {}
        self.bundle_specs: dict[str, BundleSpec] = {}
        

        self.agent_registry = get_global_agent_registry()
        self._toolname2extended_metadata: dict[str, ExtentedToolMetadata] = {}


    
    def register(
        self,
        group_doc_name: GroupName,
        bundle_spec: BundleSpec,  
        belong_to_agents: list[str], 
        bundle_role_key: BundleRoles,
        output_middleware: Callable | None = None,
        input_middleware: Callable | None = None,   
        return_direct: bool = False,
        ignore_params: list = []
    ) -> Callable:
        """
        Decorator functions
        """    

    
        available_agents_name = list(
            set(
                self.agent_registry.name2ag_conf.keys()
            )
        )

        for agent_name in belong_to_agents:
            if agent_name not in available_agents_name:
                raise ValueError(f"Agent name: {agent_name} is not available!")
        
        is_template = GROUPNAME_TO_TEMPLATE[group_doc_name]
        if is_template is None:
            raise ValueError(f"The group {group_doc_name} is not available")    
        
        if group_doc_name.value not in self.group_docs:
            template = GROUPNAME_TO_TEMPLATE[group_doc_name.value]
            group_doc = GroupDoc(
                name=group_doc_name.value,
                group_doc_template=template
            )
            self.group_docs[group_doc_name.value] = group_doc

        bundle_spec = bundle_spec
        bundle_role_key = bundle_role_key

        if bundle_spec.name not in self.bundle_specs:
            self.bundle_specs[bundle_spec.name] = bundle_spec

        

        def decorator(func: Callable):
            extended_metadata = ExtentedToolMetadata(
                func=func,
                whom_agents=belong_to_agents,
                input_middleware=input_middleware,
                output_middleware=output_middleware,
                return_direct=return_direct,
                ignore_params=ignore_params,
            )
            
            self._toolname2extended_metadata[func.__name__]  = extended_metadata
            self.group_docs[group_doc_name.value].add_tool_metadata(extended_metadata)
            self.bundle_specs[bundle_spec.name].register_tool(role_key=bundle_role_key, tool_metadata=extended_metadata)

            return func 
        return decorator
    
    def get_concrete_agent_tools(
        self,
        agent_name: str,
        **kwargs # partial params
    ) -> list[FunctionTool]:
        partial_params = kwargs
        agent_tools = list(
            filter(
                lambda x: agent_name in x[1].whom_agents, self._toolname2extended_metadata.items()
            )
        )
        function_tools = [
            tool.return_functool(partial_params=partial_params) for _, tool in agent_tools
        ]
       
        return function_tools

    def get_raw_agent_function(
        self,
        agent_name: str,
        **kwargs
    ):
        """
        Just for testing, do not use in develop
        """
        partial_params = kwargs
        agent_tools = list(
            filter(
                lambda x: agent_name in x[1].whom_agents, self._toolname2extended_metadata.items()
            )
        )
        function_tools = [
            tool.return_partial_params(partial_params=partial_params) for _, tool in agent_tools
        ]

        return function_tools
        
    
    def get_all_available_group_tools(self) -> str:

        """
        Get high-level overview of tool categories.
        
        Returns grouped tool names by functional category.
        Use this FIRST to understand what tool groups exist, then drill down with 
        get_specific_tool_documentation() for details.
        
        When to use:
        - Starting a new task and need to orient yourself
        - Don't know which tools are available
        - Want category-level overview before detailed inspection
        
        When NOT to use:
        - Already know which specific tool you need
        - Need detailed parameter schemas (use get_specific_tool_documentation)
        """

        lines = ['Available group doc ']
        for group_doc in self.group_docs.values():
            lines.append(group_doc.generate_document())
        return '\n'.join(lines)

    def get_all_available_bundle_spec(self) -> str:
        """
        Get workflow strategies and role-based tool groupings.
        
        Returns bundle specifications showing which tools work together for 
        specific workflows (e.g., "Video Evidence Worker" bundle groups search, 
        navigation, and evidence persistence tools).
        
        When to use:
        - Need to understand recommended tool workflows
        - Want to see which tools belong to your assigned role
        - Planning multi-step task execution strategy
        
        When NOT to use:
        - Just need a single tool's parameter details
        - Want simple category grouping (use get_all_available_group_tools)
        
        Note: Bundles define STRATEGY, not requirements. You can use tools 
        from different bundles if needed.
        """
        lines = ['Availabel bundle spec']
        for bundle_spec in self.bundle_specs.values():
            name = bundle_spec.name
            description = bundle_spec.description
            roles = bundle_spec.roles
            if roles:
                roles_text = "\n".join([f"  - **{str(role)}**" for role in roles])
            else:
                roles_text = "  - No roles defined."
            lines.append(f"""
            ## Bundle name: {name}

            **Description:**  
            {description}

            **Roles Included:**  
            {roles_text}

            """.strip("\n"))

        return "\n\n".join(lines)

    def list_all_tool_name(self) -> list[str]:
        """
        Get flat list of all available tool names.
        
        Returns simple list of tool names without descriptions or grouping.
        Useful for quick reference or checking if a specific tool exists.
        
        When to use:
        - Need quick lookup of available tool names
        - Checking if a tool exists in the system
        - Want minimal output without descriptions
        
        When NOT to use:
        - Need tool descriptions or parameters (use get_specific_tool_documentation)
        - Want organized grouping (use get_all_available_group_tools)
        """
        return list(self._toolname2extended_metadata.keys())

    def get_group_documentation(self, group_name: str) -> str | None:
        """
        Get detailed documentation for a specific tool group.
        
        Returns all tools in a functional category (e.g., SEARCH_GROUP, 
        CONTEXT_RETRIEVE_GROUP) with their descriptions.
        
        When to use:
        - Know the group name and want all tools in that category
        - Exploring tools within a specific functional area
        - Need more detail than get_all_available_group_tools provides
        
        When NOT to use:
        - Need individual tool's parameter schema (use get_specific_tool_documentation)
        - Don't know group names yet (use get_all_available_group_tools first)
        """

        group = self.group_docs.get(group_name)
        return group.generate_document() if group else None

    def get_bundle_documentation(self, bundle_name: str) -> str | None:
        """
        Get detailed workflow documentation for a specific bundle.
        
        Returns complete bundle spec including role definitions, tool assignments,
        workflow narrative, and typical usage patterns.
        
        When to use:
        - Assigned to a specific role
        - Need step-by-step workflow guidance for your bundle
        - Want to understand tool dependencies and execution order
        
        When NOT to use:
        - Need individual tool parameter details (use get_specific_tool_documentation)
        - Don't know bundle names (use get_all_available_bundle_spec first)
        
        Critical: This provides recommended workflow, not strict requirements.
        """
        bundle = self.bundle_specs.get(bundle_name)
        return bundle.generate_doc() if bundle else None

    def all_tool_metadata(self) -> str :
        """
        Get complete catalog of all tools with descriptions and grouping.
        
        Returns comprehensive listing showing tool name, description, and 
        which group it belongs to. Middle-ground between minimal list and 
        full documentation.
        
        When to use:
        - Need overview of ALL tools with descriptions (great for planning ahead, thinking strategies, ...)
        - Want to see tools organized by group
        - Exploring system capabilities comprehensively
        
        When NOT to use:
        - Need parameter schemas (use get_specific_tool_documentation)
        - Just need tool names (use list_all_tool_name)
        """

        lines = [f"All tool metadata: "]
        for tool_metadata in self._toolname2extended_metadata.values():
            tool_name = tool_metadata.tool_name 
            tool_description = tool_metadata.tool_description

            # find group doc
            tool_group_doc = next(
                filter(lambda x: x.is_tool_belong(tool_name), self.group_docs.values())
            )
            group_name = tool_group_doc.name if tool_group_doc else "Ungrouped"

            lines.append(f"## {group_name}")
            lines.append(f"  - {tool_name}")
            lines.append(f"    Description: {tool_description}\n")
        
        return '\n'.join(lines)
            
    def get_specific_tool_documentation(self, tool_name: str) -> str:
        """
        Get complete documentation for a single tool including parameter schema.
        
        Returns detailed specification: tool name, description, group, bundle, 
        AND full JSON schema of parameters. This is what you need BEFORE calling 
        any tool to understand its exact input requirements.
        
        When to use:
        - About to call a tool and need parameter details
        - Need to know exact parameter types and constraints
        - Want full context: what the tool does + how to use it
        - Debugging tool call errors
        
        When NOT to use:
        - Just browsing available tools (use get_all_available_group_tools)
        - Don't know tool name yet (explore groups/bundles first)
        
        CRITICAL: ALWAYS check tool documentation BEFORE calling any tool you 
        haven't used. Don't guess parameter names, types, or requirements.
        The JSON schema shows EXACTLY what parameters are required/optional.
        
        """
    
        tool_metadata = next(
            filter(lambda x: x.tool_name == tool_name, self._toolname2extended_metadata.values()) #type:ignore
        )
        if tool_metadata is None:
            return f"Toolname: {tool_name} does not existed in the system"

        tool_name = tool_metadata.tool_name if tool_metadata.tool_name else "This tool has no name"  #type:ignore
        tool_description = tool_metadata.tool_description
        tool_fn_schema = tool_metadata.tool_fn_schema_str 
        tool_group_doc = next(
            filter(lambda x: x.is_tool_belong(tool_name), self.group_docs.values())
        )
        group_name = tool_group_doc.name if tool_group_doc else "Ungrouped"

        bundle_spec = next(
            (b for b in self.bundle_specs.values() if b.recognize_tool(tool_name)),
            None
        )
        bundle_name = bundle_spec.name if bundle_spec else "No bundle assigned"

        return f"""
        ===========================
        Tool Documentation
        ===========================

        **Tool Name:**  
        {tool_name}

        **Group:**  
        {group_name}

        **Bundle Spec:**  
        {bundle_name}

        **Description:**  
        {tool_description}

        **Function Schema:**  
        {tool_fn_schema}

        """.strip()
        
tool_registry = ToolRegistry()


def get_registry_tools() -> list[FunctionTool]:
    registry_discovery_fns = [
        tool_registry.get_all_available_group_tools,
        tool_registry.get_all_available_bundle_spec,
        # tool_registry.list_all_tool_name,
        # tool_registry.get_group_documentation,
        # tool_registry.get_bundle_documentation,
        # # tool_registry.all_tool_metadata,
        # tool_registry.get_specific_tool_documentation,
    ]
    return [FunctionTool.from_defaults(fn=f) for f in registry_discovery_fns]