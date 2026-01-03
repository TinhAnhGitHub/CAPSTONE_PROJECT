from __future__ import annotations
from typing import Callable, cast
from inspect import iscoroutinefunction
import json
from functools import partial, wraps
from enum import StrEnum
from pydantic import BaseModel, Field, model_validator

from llama_index.core.tools import ToolMetadata, FunctionTool
from llama_index.core.tools.utils import create_schema_from_function
from llama_index.core.tools.function_tool import _is_context_param
from llama_index.core.bridge.pydantic import FieldInfo
import inspect

from .doc_template.group_doc import GroupName
from .middleware.wrapper import wrap_tool_with_middleware


class BundleRoles(StrEnum):
    QUERY_ANALYZER = "QUERY_ANALYZER"
    SEMANTIC_SEARCHER = "SEMANTIC_SEARCHER"
    TRANSCRIPT_ANALYZER = "TRANSCRIPT_ANALYZER"
    WORKER_RESULT_INSPECTOR = "WORKER_RESULT_INSPECTOR"
    ORCHESTRATOR_WORK_INSPECTOR = "ORCHESTRATOR_WORK_INSPECTOR"
    ORCHESTRATOR_PLANNER = 'ORCHESTRATOR_PLANNER'
    WORKER_EVIDENCE_MANAGER = "WORKER_EVIDENCE_MANAGER"
    ORCHESTRATOR_EVIDENCE_MANAGER = "ORCHESTRATOR_EVIDENCE_MANAGER"
    VIDEO_NAVIGATOR = 'VIDEO_NAVIGATOR'



class BundleRole(BaseModel):
    name: str
    description: str
    purpose: str

    typical_inputs_from: list[str] = Field(  
        default_factory=list,
        description="Which other roles usually produce inputs for this role (for documentation & agent planning)"
    )


class BundleSpec(BaseModel):
    """
    Defines a logical workflow independent of specific tools
    Acts as a container for roles and the narrative
    """    

    name: str
    description: str
    roles: dict[str, BundleRole]
    workflow_narrative: str


    registered_tools :dict[str, list[ExtentedToolMetadata ]] = Field(default_factory=dict)


    def register_tool(self, role_key: str, tool_metadata: ExtentedToolMetadata ):
        if role_key not in self.roles:
            raise ValueError(f"Role '{role_key}' is not defined in bundle '{self.name}'")

        if role_key not in self.registered_tools:
            self.registered_tools[role_key] = []
        self.registered_tools[role_key].append(tool_metadata)
    
    def recognize_tool(self, tool_name: str)  -> str | None:
        tool_group = next(
            (
                (tool_metadata, role_name)
                for role_name, tools in self.registered_tools.items()
                for tool_metadata in tools
                if tool_metadata.tool_name == tool_name #type:ignore 
            ),
            None
        )

        if tool_group is None:
            return None

        group_name = tool_group[1]
        role = self.roles[group_name]
        template = f"""
        This tool belong to Bundle role: {role.name}
        Role Description: {role.description}
        Role purpose: {role.purpose}
        """
        return template

    def generate_doc(self) -> str:

        lines = ["## Note: If there are any tools that you do not know, please use the given discovery tools to find out. You must not 'guess' the input/output, the docstring, and the signature of the tool.\n\n"]

        lines.append(f"# Strategy: {self.name}\n")
        lines.append(f"{self.description}\n\n")
       

        
        lines.append("## 1. Available Tools by Role\n")

        context = {}
        for role_key, role_def in self.roles.items():
            tools = self.registered_tools.get(role_key, [])
            if not tools:
                context[role_key]= f"[MISSING TOOL FOR {role_def.name}]"
                continue

            tool_list_name = ', '.join([cast(str, tool.tool_name) for tool in tools]) #type:ignore
            

            context[role_key] = tool_list_name
            
            lines.append(f"### 🔹 Role: {role_def.name}")
            lines.append(f"   - **Goal**: {role_def.description}")
            lines.append(f"   - The main purpose of this type of toolbox: {role_def.purpose}")
            lines.append(f"   - **Use these tools**: {tool_list_name}\n")

            lines.append(
            f"   - The group dependencies that should be execure before this group: "
            f"{', '.join(role_def.typical_inputs_from) if role_def.typical_inputs_from else 'No dependencies, use it first!'}"
)

        
        lines.append("## 2. Workflow Playbook\n")
        lines.append("To execute this strategy, follow these steps:\n")

        try: 
            formatted_workflow  = self.workflow_narrative.format(
                **context
            )
            lines.append(formatted_workflow)
        except KeyError as e:
            lines.append(f"\n*Warning: Documentation incomplete. Missing role assignment for {e}*")
        
        lines.append("\n" + "-"*50 + "\n")
        return "\n".join(lines)
    
    


def safe_partial(func: Callable, **kwargs):
    sig = inspect.signature(func)
    valid_params = sig.parameters.keys()

    filtered_kwargs = {
        k:v for k, v in kwargs.items() if k in valid_params
    }
    return partial(func, **filtered_kwargs)

def make_callable_from_partial(fn_partial: Callable) -> Callable:
    if not isinstance(fn_partial, partial):
        return fn_partial
    base_fn = fn_partial.func
    @wraps(base_fn)
    async def async_wrapper(*args, **kwargs):
        result = await fn_partial(*args, **kwargs)
        return result

    @wraps(base_fn)
    def sync_wrapper(*args, **kwargs):
        result = fn_partial(*args, **kwargs)
        return result
    
    wrapper = async_wrapper if inspect.iscoroutinefunction(base_fn) else sync_wrapper
    sig = inspect.signature(base_fn)
    bound_params = set(fn_partial.keywords.keys()) if fn_partial.keywords else set()
    new_params = [p for n, p in sig.parameters.items() if n not in bound_params]
    wrapper.__signature__ = sig.replace(parameters=new_params) #type:ignore


    wrapper.__name__ = getattr(base_fn, "__name__", "wrapped_partial")
    wrapper.__doc__ = getattr(base_fn, "__doc__", "")
    return wrapper

class ExtentedToolMetadata(BaseModel):
    whom_agents: list[str] # appear at initialization
    func: Callable


    input_middleware: Callable | None # appear at initialization
    output_middleware: Callable | None # appear at initialization
    return_direct: bool # appear at initialization
    ignore_params: list[str] = Field(default_factory=list) # appear at initialization


    tool_name: str | None = Field(default=None) # after model validator, construct based on func, ignoreparams...
    tool_description: str | None = Field(default=None) # after model validator, construct based on func, ignoreparams...
    tool_fn_schema_str: str | None = Field(default=None)

    @staticmethod
    def _get_parameters_dict(fn_schema: type[BaseModel] | None) -> dict:
        if fn_schema is None:
            parameters = {
                "type": "object",
                "properties": {
                    "input": {"title": "input query string", "type": "string"},
                },
                "required": ["input"],
            }
        else:
            parameters = fn_schema.model_json_schema()
            parameters = {
                k: v
                for k, v in parameters.items()
                if k in ["type", "properties", "required", "definitions", "$defs"]
            }
        return parameters


    @model_validator(mode="after")
    def prepare_tool_metadata(self):
        """
        Prepare the tool metadata stuff
        name
        description
        fn schema str
        -> these will be ready for the discovery tools
        """
        wrapped_functions = wrap_tool_with_middleware(
            tool=self.func,
            input_middleware=self.input_middleware,
            output_middleware=self.output_middleware,
        )

        self.func = wrapped_functions

        self.tool_name = self.func.__name__
    
        fn_sig = inspect.signature(self.func)
        fn_params = set(fn_sig.parameters.keys())
        docstring = self.func.__doc__ or ""
        param_docs, unknown_params = FunctionTool.extract_param_docs(docstring, fn_params)

        ctx_param_name = None
        has_self = False
        filtered_params = []

        for param in fn_sig.parameters.values():
            if _is_context_param(param.annotation):
                ctx_param_name = param.name
                continue
            if param.name == "self":
                has_self = True
                continue
            filtered_params.append(param)

        final_params = [
            param.replace(default=inspect.Parameter.empty)
            if isinstance(param.default, FieldInfo)
            else param
            for param in filtered_params
        ]

        fn_sig = fn_sig.replace(parameters=final_params)
        
        self.tool_description = f"{self.tool_name}{fn_sig}\n"
        if docstring:
            self.tool_description += docstring

        self.tool_description = self.tool_description.strip()


        # build fn_schema_str
        ignore_fields = self.ignore_params
        if ctx_param_name:
            ignore_fields.append(ctx_param_name)
        if has_self:
            ignore_fields.append("self")
        
        fn_schema = create_schema_from_function(
            f"{self.tool_name}",
            self.func,
            additional_fields=None,
            ignore_fields=ignore_fields,
        )

        if fn_schema is not None and param_docs:
            for param_name, field in fn_schema.model_fields.items():
                if not field.description and param_name in param_docs:
                    field.description = param_docs[param_name].strip()

        parameters = self._get_parameters_dict(fn_schema)
        self.tool_fn_schema_str = json.dumps(parameters, ensure_ascii=False)

        return self
        



    def return_functool(
        self,
        partial_params: dict
    ) -> FunctionTool:
     
        if iscoroutinefunction(self.func):
            functool = FunctionTool.from_defaults(
                async_fn=self.func,
                partial_params=partial_params
            )
        else:
            functool = FunctionTool.from_defaults(
                fn=self.func,
                partial_params=partial_params
            )

        return functool
    


    def return_partial_params(
        self, partial_params: dict
    ):
      

        partial_function = safe_partial(self.func, **partial_params)
        return make_callable_from_partial(fn_partial=partial_function)
         

        

        




