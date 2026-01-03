import inspect
import textwrap
from typing import Annotated, Callable, Awaitable, Any, get_origin, get_args, get_type_hints
from functools import wraps

from videodeepsearch.tools.base.schema import ImageInterface, SegmentInterface




def annotation_matches(value: Any, annotation: Any) -> bool:
    """Return True if `value` is an instance of `annotation`, handling generics."""
    if annotation is Any or annotation is inspect._empty:
        return False

    origin = get_origin(annotation)
    if origin:
        return isinstance(value, origin)

    return isinstance(value, annotation)


def types_compatible(source_type: Any, target_type: Any) -> bool:
    """Whether `source_type` should be injected into a parameter of `target_type`."""
    if source_type is Any or target_type is Any:
        return False
    if source_type is inspect._empty or target_type is inspect._empty:
        return False

    if get_origin(source_type) is Annotated:
        source_type = get_args(source_type)[0]
    if get_origin(target_type) is Annotated:
        target_type = get_args(target_type)[0]
        
    if source_type == target_type:
        return True
    
    src_origin = get_origin(source_type)
    tgt_origin = get_origin(target_type)



    if src_origin and tgt_origin:
        if not issubclass(src_origin, tgt_origin):
            return False
        
        src_args = get_args(source_type)
        tgt_args = get_args(target_type)

        if not tgt_args:
            return True
        
        if len(src_args) != len(tgt_args):
            return False
        
        return all(types_compatible(s, t) for s,t in zip(src_args, tgt_args))

    eff_src = src_origin if src_origin else source_type
    eff_tgt = tgt_origin if tgt_origin else target_type

    try:
        return issubclass(eff_src, eff_tgt)
    except Exception as e:
        return False
        

def wrap_tool_with_middleware(
    tool: Callable[..., Awaitable[Any]],
    *,
    input_middleware: Callable[..., Awaitable[Any]] | None = None,
    output_middleware: Callable[..., Awaitable[Any]] | None = None,
):
    tool_sig = inspect.signature(tool)
    tool_hints = get_type_hints(tool, include_extras=True)
    tool_return = tool_hints.get('return', Any)


    input_sig = inspect.signature(input_middleware) if input_middleware else None
    input_hints = get_type_hints(input_middleware, include_extras=True) if input_middleware else {}
    input_return = input_hints.get('return', Any) if input_middleware else None



    output_sig = inspect.signature(output_middleware) if output_middleware else None
    output_hints = get_type_hints(output_middleware, include_extras=True) if output_middleware else {}
    output_return = output_hints.get('return', Any) if output_middleware else None

    
    public_params = []

    if input_sig:
        public_params.extend(input_sig.parameters.values())

    for name, param in tool_sig.parameters.items():
        annotation =  input_hints.get(name, param.annotation)
        if input_middleware and types_compatible(input_return, annotation):
            continue  
        public_params.append(param)
  
    if output_sig:
        for name, param in output_sig.parameters.items():
            annotation = output_hints.get(name, param.annotation)
           
            if types_compatible(tool_return, annotation):
                continue  

            public_params.append(param)

    final_return = output_return if output_middleware else tool_return

    new_signature = inspect.Signature(parameters=public_params, return_annotation=final_return)

   
    @wraps(tool)
    async def wrapper(**kwargs) -> Any:
        
        injected_input_value = None

        if input_middleware and input_sig:
            input_kwargs = {
                name: kwargs[name]
                for name in input_sig.parameters.keys()
                if name in kwargs
            }
            injected_input_value = await input_middleware(**input_kwargs)

     
        tool_kwargs = {}

        for name, param in tool_sig.parameters.items():
            annotation = param.annotation

            if input_middleware and types_compatible(input_return, annotation):
                tool_kwargs[name] = injected_input_value
                continue

            if name in kwargs:
                tool_kwargs[name] = kwargs[name]

     
        result = await tool(**tool_kwargs)

        if not output_middleware or not output_sig:
            return result

       
        output_kwargs = {}

        for name, param in output_sig.parameters.items():
            annotation = param.annotation

            if types_compatible(tool_return, annotation):
                output_kwargs[name] = result
                continue

            if name in kwargs:
                output_kwargs[name] = kwargs[name]

        return output_middleware(**output_kwargs)

    wrapper.__signature__ = new_signature #type:ignore

    
    tool_doc = (tool.__doc__ or "").strip()
    input_doc = (input_middleware.__doc__ or "").strip() if input_middleware else ""
    output_doc = (output_middleware.__doc__ or "").strip() if output_middleware else ""

    if not input_doc and not output_doc:
        wrapper.__doc__ = textwrap.dedent(tool_doc).strip()
        return wrapper

    middleware_note = f"""
    This is the middlware tool wrapper! Please notice the input/output middleware doc!
    This is the original tool docstring:
    {tool_doc}

    This is the input middleware: Please notice that if the input middleware is wrapped to this tool, you just need to fill in the input's middleware params (auto provided). And the middleware will automatically resolve the wrapped function's params so that it is compatible to the tool.
    Input middleware doc:
    {input_doc if input_doc else "No input middleware"}

    This is the output middleware: Please notice that if the ouptut middleware is wrapped to this tool, it will automatically resolve the wrapped function's output. It should return a DataHandle. Use the information in the handle id coupled with the viewing set of tools (it should have compatible param's name, you would know). 
    Output middleware doc:
    {output_doc if output_doc else "No output middleware!"}
    """

    wrapper.__doc__ = textwrap.dedent(middleware_note).strip()

    return wrapper
