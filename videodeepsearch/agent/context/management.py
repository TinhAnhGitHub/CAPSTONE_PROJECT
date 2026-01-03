import os
import json
from pathlib import Path
from pydantic import BaseModel
from typing import TypeVar, Optional, List
from llama_index.core.workflow import Context, JsonPickleSerializer

T = TypeVar('T', bound=BaseModel)

class FileSystemContextStore:
    def __init__(self, storage_dir: str = 'workflow_storage'):
        self.storage_path = Path(storage_dir)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, key: str) -> Path:
        safe_key = "".join([c for c in key if c.isalnum() or c in ('-', '_')])
        return self.storage_path / f"{safe_key}.json"
    
    def save_context(self, session_id: str, context_model: Context):
        
        context_dict = context_model.to_dict(JsonPickleSerializer())
        file_path = self._get_file_path(session_id)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(context_dict, f, indent=4, ensure_ascii=False)
        
        print(f"Context saved for session: {session_id}")
    
    def load_context(self, session_id: str) -> dict | None:
        file_path = self._get_file_path(session_id)
        print(f"{file_path=}")
        if not file_path.exists():
            return None 
        
        with open(file_path, 'r', encoding='utf-8') as f:
            context_dict = json.load(f)
        return context_dict
        
        


        
        