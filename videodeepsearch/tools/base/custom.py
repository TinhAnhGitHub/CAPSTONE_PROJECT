from llama_index.core.tools.utils import create_schema_from_function
from llama_index.core.tools import FunctionTool, ToolMetadata
from llama_index.core.tools.function_tool import AsyncCallable, _is_context_param
import inspect
from typing import Optional, Callable, Type, Any, Dict
from llama_index.core.bridge.pydantic import BaseModel, FieldInfo


class ExtendedFunctionTool(FunctionTool):
    @classmethod
    def from_defaults_extended(
        cls,
        fn: Optional[Callable[..., Any]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        fn_schema: Optional[Type[BaseModel]] = None,
        async_fn: Optional[AsyncCallable] = None,
        tool_metadata: Optional[ToolMetadata] = None,
        callback: Optional[Callable[[Any], Any]] = None,
        async_callback: Optional[AsyncCallable] = None,
        partial_params: Optional[Dict[str, Any]] = None,
        ignore_params: list[str] = []
    ) -> "FunctionTool":
        
        partial_params = partial_params or {}

        if tool_metadata is None:
            fn_to_parse = fn or async_fn
            assert fn_to_parse is not None, "fn must be provided"
            name = name or fn_to_parse.__name__
            docstring = fn_to_parse.__doc__ or ""

            # Get function signature
            fn_sig = inspect.signature(fn_to_parse)
            fn_params = set(fn_sig.parameters.keys())

            # 1. Extract docstring param descriptions
            param_docs, unknown_params = cls.extract_param_docs(docstring, fn_params)

            # 2. Filter context and self in a single pass
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

            # 3. Remove FieldInfo defaults and partial_params
            final_params = [
                param.replace(default=inspect.Parameter.empty)
                if isinstance(param.default, FieldInfo)
                else param
                for param in filtered_params
                if param.name not in (partial_params or {})
            ]

            # 4. Replace signature in one go
            fn_sig = fn_sig.replace(parameters=final_params)

            # 5. Build description
            if description is None:
                description = f"{name}{fn_sig}\n"
                if docstring:
                    description += docstring

                description = description.strip()

            # 6. Build fn_schema only if not already provided
            if fn_schema is None:
                
                if ctx_param_name:
                    ignore_params.append(ctx_param_name)
                if has_self:
                    ignore_params.append("self")
                ignore_params.extend(partial_params.keys())

                ignore_params = list(set(ignore_params))

                fn_schema = create_schema_from_function(
                    f"{name}",
                    fn_to_parse,
                    additional_fields=None,
                    ignore_fields=ignore_params,
                )
                if fn_schema is not None and param_docs:
                    for param_name, field in fn_schema.model_fields.items():
                        if not field.description and param_name in param_docs:
                            field.description = param_docs[param_name].strip()

            tool_metadata = ToolMetadata(
                name=name,
                description=description,
                fn_schema=fn_schema,
                return_direct=return_direct,
            )
        return cls(
            fn=fn,
            metadata=tool_metadata,
            async_fn=async_fn,
            callback=callback,
            async_callback=async_callback,
            partial_params=partial_params,
        )
        