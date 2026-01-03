from typing import Callable, cast
import inspect
from functools import partial
from llama_index.core.tools import FunctionTool
from .registry import tool_registry

def safe_partial(func: Callable, **kwargs):
    sig = inspect.signature(func)
    valid_params = sig.parameters.keys()

    filtered_kwargs = {
        k:v for k, v in kwargs.items() if k in valid_params
    }
    return partial(func, **filtered_kwargs)


def get_all_tools_normal(
        self, **kwargs
    )->dict[str, Callable]:
        fn_name2fn_tool = {}
        for name in tool_registry.list_all_tool_name():
            if self._bind_tool(name) is not None:
                fnc = cast(Callable, self._bind_tool(name))
                fnc = safe_partial(fnc, **kwargs)
                fnc = self._make_callable_from_partial(fnc, None)               
                fn_name2fn_tool[fnc.__name__] = fnc
        return fn_name2fn_tool


def get_all_tools_functool(
        self,
        **kwargs
    ) -> dict[str, FunctionTool]:
        """
        Turn the tools into FunctionToosl
        If kwargs provide -> add to the partial functions
        """
        fn_name2fn_tool = {}
        for name in tool_registry.list_all_tool_name():
            fn = self._bind_tool(name)

            if fn is not None:
                return_type = self._get_return_type(tool_registry.get(name=name).func) #type:ignore
                formatter = self.formatter.get_formatter(return_type=return_type)

                fnc = cast(Callable, fn)
                fnc = safe_partial(fnc, **kwargs)
                fnc = self._make_callable_from_partial(fnc, formatter)


                if inspect.iscoroutinefunction(fnc):
                    func_tool = FunctionTool.from_defaults(
                        async_fn=fnc
                    )
                else:
                    func_tool = FunctionTool.from_defaults(
                        fn=fnc
                    )                
                fn_name2fn_tool[fnc.__name__] = func_tool

        return fn_name2fn_tool