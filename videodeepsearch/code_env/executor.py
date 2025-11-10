import ast
import nest_asyncio
import contextlib
import io
import asyncio
import textwrap
import dataclasses
import traceback
from typing import Callable, Any, Iterable, Mapping, Sequence, Coroutine
from llama_index.core.agent import CodeActAgent
import sys


class TeeIO(io.StringIO):
    def __init__(self, original_stream):
        super().__init__()
        self._original = original_stream

    def write(self, s):
        self._original.write(s)
        self._original.flush()
        return super().write(s)

    def flush(self):
        self._original.flush()
        return super().flush()
    

_SAFE_BUILTINS: dict[str, Any] =  {
    
    'abs': abs, 'divmod': divmod, 'max': max, 'min': min,
    'pow': pow, 'round': round, 'sum': sum,

    # Type Conversions
    'bool': bool, 'complex': complex, 'float': float, 'int': int,
    'str': str, 'tuple': tuple, 'list': list, 'dict': dict,
    'set': set, 'frozenset': frozenset, 'range': range,
    'bytes': bytes, 'bytearray': bytearray,

    # Functional / Iteration
    'enumerate': enumerate, 'filter': filter, 'map': map,
    'zip': zip, 'sorted': sorted, 'reversed': reversed,
    'all': all, 'any': any,

    # Object / Introspection (safe ones)
    'len': len, 'hash': hash, 'id': id,
    'isinstance': isinstance, 'issubclass': issubclass,
    'type': type, 'dir': dir, 'vars': vars, 'callable': callable,

    # String / Representation
    'chr': chr, 'ord': ord, 'format': format,
    'repr': repr, 'ascii': ascii,

    # Miscellaneous
    'object': object, 'slice': slice,
    'staticmethod': staticmethod, 'classmethod': classmethod,
    'property': property,
    'print': print
}




class SandboxViolationError(RuntimeError):
    """Raise if the code exec have some violations"""

@dataclasses.dataclass(slots=True)
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    return_value: Any | None
    exception_repr: str | None

    def to_message(self) -> str:
        parts: list[str] = []
        if self.stdout:
            parts.append(f"stdout:\n{textwrap.dedent(self.stdout).rstrip()}")
        if self.stderr:
            parts.append(f"stderr:\n{textwrap.dedent(self.stderr).rstrip()}")
        if self.exception_repr:
              parts.append(f"exception:\n{self.exception_repr.rstrip()}")
        if self.return_value is not None:
            parts.append(f"return_value:\n{self.return_value!r}")
        if not parts:
            parts.append("✅ execution succeeded with no output")
        return "\n\n".join(parts)
    
class SandboxCodeExecutor:
    """
    Stateful, restricted env execution
    """
    def __init__(
        self, 
        *, 
        shared_bindings: dict[str, Any] | None = None,
        allowed_builtins: dict[str, Any] | None = None,
        allowed_modules: Sequence[str] | None = None,
        persist_state: bool = True,
    ):
        self._persist_state = persist_state

        safe_builtins = dict(_SAFE_BUILTINS)
        if allowed_builtins:
            safe_builtins.update(allowed_builtins)

        self._allowed_modules = allowed_modules
        
        def _sandbox(
            name: str,
            globals: Mapping[str, Any] | None = None,
            locals: Mapping[str, Any] | None = None,
            fromlist: Iterable[str] = (),
            level:int = 0
        ):
            root_name: str = name.split('.', 1)[0]
            
            if allowed_modules:
                if root_name not in allowed_modules:
                    raise SandboxViolationError(f"Import of module {name} is not allowed. Allowed: {allowed_modules}")
            return __import__(name, globals, locals, tuple(fromlist), level)

        safe_builtins["__import__"] = _sandbox
        
        self._base_globals = {
            '__builtins__': safe_builtins
        }
        if shared_bindings:
            self._base_globals.update(shared_bindings)
        
        self._globals = dict(self._base_globals)
        self._locals = {}
    

    
    
    def execute(self, code: str) -> ExecutionResult:
        stdout_buf = TeeIO(sys.stdout)
        stderr_buf = TeeIO(sys.stderr)

        global_ctx = self._globals if self._persist_state else dict(self._base_globals)
        local_ctx = self._locals if self._persist_state else {}

        stripped = textwrap.dedent(code).strip('\n')
        return_value: Any | None = None
        exception_repr: str | None = None

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                tree = ast.parse(stripped)
                
                has_await = any(
                    isinstance(node, ast.Await) for node in ast.walk(tree)
                )
                if has_await:
                    wrapped_code = f"async def __sandbox_async__():\n{textwrap.indent(stripped, '    ')}"
                    exec(compile(wrapped_code, "<sandbox>", "exec"), global_ctx, local_ctx)
                    coro = local_ctx["__sandbox_async__"]()

                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None

                    if loop and loop.is_running():
                        nest_asyncio.apply()
                        return_value = asyncio.ensure_future(coro)
                        return_value = loop.run_until_complete(asyncio.gather(return_value))[0] #type:ignore
                    else:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        return_value = loop.run_until_complete(coro)
                        loop.close()
                else:
                
                    last_expr = (
                        tree.body[-1] if tree.body  and isinstance(tree.body[-1], ast.Expr) else None
                    )
                    if last_expr is not None:
                        prefix = tree.body[:-1]
                        if prefix:
                            exec(compile(ast.Module(prefix, []), "<sandbox>", "exec"), global_ctx, local_ctx)
                            return_value = eval(
                                compile(ast.Expression(last_expr.value), "<sandbox>", "eval"), global_ctx, local_ctx 
                            )
                        else:
                            exec(compile(tree, '<sandbox>', 'exec'))
        except SandboxViolationError as e:
            exception_repr = f"SandboxViolation Error: {e}"
        
        except Exception as exc:
            tb = traceback.format_exc()
            exception_repr = f"{type(exc).__name__}: {exc}\n{tb}"
        
        stdout = stdout_buf.getvalue()
        stderr = stderr_buf.getvalue()

        success = exception_repr is None
        return ExecutionResult(
            success=success,
            stdout=stdout,
            stderr=stderr,
            return_value=return_value if success else None,
            exception_repr=exception_repr
        )
    

def build_worker_executor(
    *,
    tools_bindings: dict[str, Callable[..., Any]],
    extra_globals: dict[str, Any] | None = None,
    allowed_modules: Sequence[str] | None = None
)-> Callable[[str], Coroutine[Any, Any, str]]:
    shared_bindings = {}
    for name, fn in tools_bindings.items():
        shared_bindings[name] = fn
    
    if extra_globals:
        shared_bindings.update(extra_globals)
    
    executor = SandboxCodeExecutor(
        shared_bindings=shared_bindings,
        allowed_modules=allowed_modules
    )
    async def _run(code:str)->str:
        result = executor.execute(code)
        return result.to_message()

    return _run

    


    
