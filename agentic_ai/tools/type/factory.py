from llama_index.core.tools import FunctionTool
from functools import partial, wraps
import json
from collections import defaultdict
from typing import Callable, Annotated, get_origin, get_args, get_type_hints, cast
import inspect

# sus
from agentic_ai.tools.clients.milvus.client import ImageMilvusClient, SegmentCaptionImageMilvusClient
from agentic_ai.tools.clients.postgre.client import PostgresClient
from agentic_ai.tools.clients.minio.client import StorageClient
from agentic_ai.tools.clients.external.encode_client import ExternalEncodeClient
from agentic_ai.tools.schema.artifact import VideoInterface, ImageObjectInterface, SegmentObjectInterface

from llama_index.core.llms import LLM


from llama_index.core.llms import  TextBlock, ImageBlock, VideoBlock
from llama_index.core.base.llms.types import ContentBlock

from llama_index.core.tools import FunctionTool

from .registry import tool_registry


def safe_partial(func: Callable, **kwargs):
    sig = inspect.signature(func)
    valid_params = sig.parameters.keys()

    filtered_kwargs = {
        k:v for k, v in kwargs.items() if k in valid_params
    }
    return partial(func, **filtered_kwargs)




class ToolOutputFormatter:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._formatters: dict[type, Callable] = {
            list[ImageObjectInterface]: self.format_image_interface,
            list[SegmentObjectInterface]: self.format_segment_interface,
            list[tuple[bytes, str]]: self.format_bytes_object,
            list[VideoInterface]: self.format_video_interface
            
        }
        self._initialized = True
    
    def get_formatter(self, return_type: type) -> Callable | None:
        if return_type in self._formatters:
            return self._formatters[return_type]        
        origin = get_origin(return_type)
        if origin is list:
            args = get_args(return_type)
            if args:
                list_type = list[args[0]]
                if list_type in self._formatters:
                    return self._formatters[list_type]
        
        return None
    
    @staticmethod
    def format_image_interface(
        list_images: list[ImageObjectInterface]
    ) -> ContentBlock:
        """Format list of ImageObjectInterface to structured JSON ContentBlock."""
        result_dict: dict[str, list[dict]] = defaultdict(list)
    
        for image in list_images:
            result_dict[image.related_video_id].append(image.model_dump())
        readable_result = {
            "type": "Search Image Results",
            "summary": f"Retrieved {len(list_images)} visually similar frames across {len(result_dict)} videos.",
            "results": result_dict
        }
        
        return TextBlock(text=json.dumps(readable_result, indent=2,  ensure_ascii=False))

    @staticmethod
    def format_segment_interface(
        list_segments: list[SegmentObjectInterface]
    ) -> ContentBlock:
        """Format list of SegmentObjectInterface to structured JSON ContentBlock."""
        result_dict: dict[str, list[dict]] = defaultdict(list)

        for segment in list_segments:
            result_dict[segment.related_video_id].append(segment.model_dump())

        for video_id, segments in result_dict.items():
            segments.sort(
                key=lambda x: (x.get("score") or 0.0), 
                reverse=True
            )

        readable_result = {
            "type": "visual_segment_search_result",
            "summary": (
                f"Retrieved {len(list_segments)} relevant segments across "
                f"{len(result_dict)} videos based on semantic similarity to the query."
            ),
            "results": result_dict,
        }

        return TextBlock(
            text=json.dumps(readable_result, indent=2, ensure_ascii=False)
        )

    @staticmethod
    def format_bytes_object(
        byte_mime_type: list[tuple[bytes,str]]
    ) -> list[ContentBlock]:

        result: list[ContentBlock] = []
        for byte_object, mime_type  in byte_mime_type:
            if 'image' in mime_type:
                result.append(ImageBlock(image=byte_object, image_mimetype=mime_type))
            elif 'video' in mime_type:
                result.append(VideoBlock(video=byte_object, video_mimetype=mime_type))
            else:
                raise ValueError(f"Unsupported mime_type: {mime_type}. Must contain 'image' or 'video'.")
        return result

    @staticmethod
    def format_video_interface(
        list_videos: list[VideoInterface]
    )->list[ContentBlock]:
        result = []
        for video in list_videos:
            result.append(
                TextBlock(text=str(video))
            )
        return result



class ToolFactory:
    """
    Factory -> bind dependencies to registered tools
    Use registry + explicit dependency binding for control
    """
    def __init__(
        self,
        image_milvus_client: ImageMilvusClient,
        segment_milvus_client: SegmentCaptionImageMilvusClient,
        postgres_client: PostgresClient,
        minio_client: StorageClient,
        external_client: ExternalEncodeClient,
        llm_as_tools: LLM,
    ):
        self.dependency_map = {
            'visual_milvus_client': image_milvus_client,
            'segment_milvus_client': segment_milvus_client,
            'postgres_client': postgres_client,
            'minio_client': minio_client,
            'external_client': external_client,
            'llm_as_tools': llm_as_tools
        }

        self._tool_cache: dict[str, Callable] = {}
        self.formatter =  ToolOutputFormatter()
    
    def _validate_type_annotations(self, func: Callable) -> None:
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        if 'return' not in type_hints:
            raise ValueError(
                f"Function '{func.__name__}' must have a return type annotation. "
                f"Add '-> ReturnType' to the function signature."
            )
    
        missing_annotations = []
        for param_name, param in sig.parameters.items():
            if param_name not in type_hints and param.annotation == inspect.Parameter.empty:
                missing_annotations.append(param_name)
        
        if missing_annotations:
            raise ValueError(
                f"Function '{func.__name__}' has parameters without type annotations: {missing_annotations}. "
                f"All parameters must have type annotations (use Annotated for descriptions)."
            )

    def _get_return_type(self, func: Callable)->type:
        type_hints = get_type_hints(func)
        return_type = type_hints['return']
        if get_origin(return_type) is Annotated:
            return_type = get_args(return_type)[0]
        return return_type
    
    def _make_callable_from_partial(self, fn_partial: Callable, formatter_func: Callable | None) -> Callable:
        if not isinstance(fn_partial, partial):
            return fn_partial
        base_fn = fn_partial.func
        @wraps(base_fn)
        async def async_wrapper(*args, **kwargs):
            result = await fn_partial(*args, **kwargs)
            if formatter_func:
                return formatter_func(result)
            return result

        @wraps(base_fn)
        def sync_wrapper(*args, **kwargs):
            result = fn_partial(*args, **kwargs)
            if formatter_func:
                return formatter_func(result)
            return result
        
        wrapper = async_wrapper if inspect.iscoroutinefunction(base_fn) else sync_wrapper
        sig = inspect.signature(base_fn)
        bound_params = set(fn_partial.keywords.keys()) if fn_partial.keywords else set()
        new_params = [p for n, p in sig.parameters.items() if n not in bound_params]
        wrapper.__signature__ = sig.replace(parameters=new_params)


        wrapper.__name__ = getattr(base_fn, "__name__", "wrapped_partial")
        wrapper.__doc__ = getattr(base_fn, "__doc__", "")
        return wrapper


    def _bind_tool(self, tool_name:str) -> Callable | None:
        if tool_name in self._tool_cache:
            return self._tool_cache[tool_name]

        metadata = tool_registry.get(tool_name)
        if not metadata:
            return None
        
        bound_kwargs = {}
        missing_deps = []

        for dep_name in metadata.dependencies:
            if dep_name in self.dependency_map:
                bound_kwargs[dep_name] = self.dependency_map[dep_name]
            else:
                missing_deps.append(dep_name)
        
        if missing_deps:
            raise ValueError(
                f"Tool '{tool_name}' requires dependencies {missing_deps} "
                f"but they were not provided to ToolFactory"
            )
        partial_func= partial(metadata.func, **bound_kwargs)
        print(metadata.name)
        return_type = self._get_return_type(metadata.func)
        formatter_func = self.formatter.get_formatter(return_type)
        wrapper_func = self._make_callable_from_partial(partial_func, formatter_func)
        self._tool_cache[tool_name] = wrapper_func
        return wrapper_func
    
    
    def get_tool(self, name: str) -> Callable | None:
        return self._bind_tool(name)

    def get_tools_by_category(self, category:str)->dict[str, Callable | None]:
        tool_names = tool_registry.list_by_category(category)
        return {
            name: self._bind_tool(name)
            for name in tool_names
            if self._bind_tool(name) is not None
        }
    
    def get_tools_by_tags(self, tags: list[str]) -> dict[str, Callable | None]:
        """Get tools with specific tags"""
        tool_names = tool_registry.list_by_tags(tags)
        return {
            name: self._bind_tool(name)
            for name in tool_names
            if self._bind_tool(name) is not None
        }
    
    def get_tools_by_names(self, names: list[str]) -> dict[str, Callable | None]:
        return {
            name: self._bind_tool(name)
            for name in names
            if self._bind_tool(name) is not None
        }
    

    def get_all_tools(self) -> dict[str, Callable | None]:
        return {
            name: self._bind_tool(name)
            for name in tool_registry.list_all()
            if self._bind_tool(name) is not None
        }
    
    

    def get_all_tools_functool(
        self,
        **kwargs
    ) -> dict[str, FunctionTool]:
        """
        Turn the tools into FunctionToosl
        If kwargs provide -> add to the partial functions
        """
        fn_name2fn_tool = {}
        for name in tool_registry.list_all():
            if self._bind_tool(name) is not None:
                fnc = cast(Callable, self._bind_tool(name))
                fnc = safe_partial(fnc, **kwargs)
                fnc = self._make_callable_from_partial(fnc, None)

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
    
    def get_all_tools_normal(
        self, **kwargs
    )->dict[str, Callable]:
        fn_name2fn_tool = {}
        for name in tool_registry.list_all():
            if self._bind_tool(name) is not None:
                fnc = cast(Callable, self._bind_tool(name))
                fnc = safe_partial(fnc, **kwargs)
                fnc = self._make_callable_from_partial(fnc, None)               
                fn_name2fn_tool[fnc.__name__] = fnc
        return fn_name2fn_tool
        
