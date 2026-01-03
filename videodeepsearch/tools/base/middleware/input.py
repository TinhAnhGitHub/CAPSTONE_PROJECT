from typing import cast, Annotated
from llama_index.core.workflow import Context

from videodeepsearch.agent.context.worker_context import SmallWorkerContext

from videodeepsearch.tools.clients.milvus.client import (
    ImageInterface,
    SegmentInterface
)

from .arg_doc import HANDLE_ID_ANNOTATION


async def input_simple_images_middleware(
    ctx: Context,
    agent_name: str,
    handle_id: HANDLE_ID_ANNOTATION,
    image_minio_path: Annotated[int, "The specific image minio path you want to use."]
) -> ImageInterface:
    """
    Which ever tool is wrapped with this input middleware, it means that this tools is based on the previous tool results. It needs handle_id from the previous tool's output (in which these tools are wrapped with ouptut middleware). Also it needs the image minio path (important). 
    """
    context_dict = await ctx.store.get(agent_name)
    local_agent_context = SmallWorkerContext.model_validate(context_dict)
    result_store = local_agent_context.raw_result_store
    
    data_handle = result_store.retrieve(handle_id=handle_id)
    if data_handle is None:
        raise ValueError(f"Handle tool id {handle_id} not found in ResultStore of agent: {agent_name}. Please refer to the documentation tool set to get your previous tool's result.")

    raw_data = data_handle.get_data()
    if isinstance(raw_data, ImageInterface):
        raw_data = [raw_data]

    raw_data = cast(list[ImageInterface], raw_data)

    return_data = next(
        filter(
            lambda x: x.minio_path == image_minio_path, raw_data
        )
    )
    if return_data is None:
        raise ValueError(f"The image path: {image_minio_path} does not exists in the handle: {handle_id}. Please inspect this handle data first to get the image minio path.")

    return return_data
    

# async def input_simple_segments_middleware(
    
#     segment_minio_path:  Annotated[int, "The specific segment minio path you want to use."]
# ) -> SegmentInterface:
#     """
#     Which ever tool is wrapped with this input middleware, it means that this tools is based on the previous tool results. It needs handle_id from the previous tool's output (in which these tools are wrapped with ouptut middleware). Also it needs the segment minio path (important). 
#     """
    
#     context_dict = await ctx.store.get(agent_name)
#     local_agent_context = SmallWorkerContext.model_validate(context_dict)
#     result_store = local_agent_context.raw_result_store
    
#     data_handle = result_store.retrieve(handle_id=handle_id)
#     if data_handle is None:
#         raise ValueError(f"Handle tool id {handle_id} not found in ResultStore of agent: {agent_name}")

#     raw_data = data_handle.get_data()
#     if isinstance(raw_data, SegmentInterface):
#         raw_data = [raw_data]
    
#     raw_data = cast(list[SegmentInterface], raw_data)

#     return_data = next(
#         filter(
#             lambda x: x.minio_path ==  segment_minio_path, raw_data
#         )
#     )

#     if return_data is None:
#         raise  ValueError(f"The segment path: {segment_minio_path} does not exists in the handle: {handle_id}. Please inspect this handle data first to get the segment minio path.")

#     return return_data
