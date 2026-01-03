"""
videodeepsearch/tools/base/middleware/data_handle.py
Contains 3 classes:
- DataHandle: The interface representations what the agent will see and some information
- PersistedResult: Contain the raw data, and the DataHandle
- ResultStore: Act as a way to store the results, and will be persisted in the LlamaIndex Context class.
- Only works with BaseModel I/O, since the results can be token-consuming. If tools return primitives, then just return duh
"""

from pydantic import BaseModel, Field, PrivateAttr
from typing import TypeVar, Generic, Sequence
from uuid import uuid4

from llama_index.core.agent.workflow import ToolCall

from videodeepsearch.tools.base.schema import BaseInterface


T = TypeVar("T", bound=BaseInterface | int | str | float | bool | Sequence[BaseInterface])
 
class DataHandle(BaseModel, Generic[T]):

    handle_id: str = Field(default_factory=lambda: str(uuid4()))
    tool_used: ToolCall | None = Field(None)
    summary: dict = Field(
        default_factory=dict,
        description="Human-readable summary for the agent"
    )

    related_video_ids: list[str]
    
    _raw_data: T | None = PrivateAttr(default=None)

    def set_data(self, data: T) -> None:
        self._raw_data = data
    
    def get_data(self) -> T:
        if self._raw_data is None:
            raise ValueError(
                f"DataHandle {self.handle_id} has no raw data attached. "
                f"This should not happen - middleware must call set_data()."
            )
        return self._raw_data
    
    def has_data(self) -> bool:     
        return self._raw_data is not None
    
    # def __str__(self) -> str:
    #     return (
    #         f"DataHandle("
    #         f"handle_id={self.handle_id}, "
    #         f"tool_used={self.tool_used}, "
    #         f"summary={self.summary}, "
    #         f"related_video_ids={self.related_video_ids}"
    #         f")"
    #     )




# class PersistResult(Generic[T]): 
#     """
#     Wraps a tool call result + decoded typed payload.
#     """

#     def __init__(
#         self,
#         data_handle: DataHandle[T],
#         tool_result: ToolCallResult,
#         resolved_output: T,
#     ):
#         self.data_handle = data_handle
#         self.tool_result = tool_result
#         self.resolved_output = resolved_output  # already parsed into T

#     @property
#     def result_output(self) -> T:
#         """Return the typed output."""
#         return self.resolved_output



class ResultStore(BaseModel):
    def __init__(self):
        self._store: dict[str, DataHandle] = {}

    def persist_raw(
        self,
        tool_call: ToolCall,
        summary: dict,
        parsed_output: T, #type:ignore
        related_video_ids: list[str] ,
    ) -> str:
        try:
            data_handle = DataHandle(
                summary=summary,
                related_video_ids=related_video_ids,
                tool_used=tool_call
            )
            data_handle.set_data(parsed_output)

            self._store[data_handle.handle_id] = data_handle
            return data_handle.handle_id
        except Exception as e:
            raise ValueError(f"Exception while storing results: {e}")
        
    def get_all_handle_str(self) -> str:
        ids = "\n".join(str(handle.handle_id) for handle in self._store.values())
        return f"Here is the available handle id(s): {ids}"


    def persist_handle(
        self,
        data_handle: DataHandle
    ):
        try:
            self._store[data_handle.handle_id] = data_handle
            return data_handle.handle_id
        except Exception as e:
            raise ValueError(f"Exception while storing results: {e}")
    def retrieve(self, handle_id: str) -> DataHandle | None:
        return self._store.get(handle_id)




    