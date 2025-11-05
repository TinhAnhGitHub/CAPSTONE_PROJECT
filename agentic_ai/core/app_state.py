from typing import Any
from tools.type.registry import FunctionRegistry

class Appstate:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    tool_registry: FunctionRegistry = None # type:ignore
