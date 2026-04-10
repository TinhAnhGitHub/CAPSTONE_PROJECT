import functools
import inspect
import time
from typing import ParamSpec, TypeVar, cast
from collections.abc import Awaitable
from collections.abc import Callable

import mlflow
from mlflow.entities import SpanType
from loguru import logger


P = ParamSpec('P')
R = TypeVar('R')

def traced_tool():
    def decorator(
        func: Callable[P, R] | Callable[P, Awaitable[R]]
    ) -> Callable[P, R] | Callable[P, Awaitable[R]]:

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tool_name = func.__name__
            start_time = time.time()

            with mlflow.start_span(
                name=f"tool:{tool_name}", span_type=SpanType.TOOL
            ) as span:
                result = None

                try:
                    result = await cast(Awaitable[R], func(*args, **kwargs))
                except Exception as e:
                    status = 'error'
                    span.set_attribute('status', "error")
                    span.set_attribute('error_type', type(e).__name__)
                    span.set_attribute('error_message', str(e))
                    logger.error(f"Tool {tool_name=}: Error: {e=}")
                    raise
            return result

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tool_name = func.__name__
            start_time = time.time()

            with mlflow.start_span(
                name=f"tool:{tool_name}", span_type=SpanType.TOOL
            ) as span:
                result = None

                try:
                    result = cast(R, func(*args, **kwargs))
                except Exception as e:
                    status = 'error'
                    span.set_attribute('status', "error")
                    span.set_attribute('error_type', type(e).__name__)
                    span.set_attribute('error_message', str(e))
                    logger.error(f"Tool {tool_name=}: Error: {e=}")
                    raise

            return result

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator