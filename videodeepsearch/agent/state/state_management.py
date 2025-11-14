from pathlib import Path
from dataclasses import dataclass
from llama_index.core.workflow import Context, JsonSerializer, Workflow
import json


def _sanitize_for_fs(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum() or ch in ("-", "_")) or "unknown"

@dataclass(frozen=True)
class SessionKey:
    user_id: str
    session_id: str

    def to_path(self, base_dir: Path) -> Path:
        # basedir/ user id dir / sesssion id dir
        clean_ud = _sanitize_for_fs(self.user_id)
        clean_ss = _sanitize_for_fs(self.session_id)
        return base_dir / clean_ud / f"{clean_ss}.json"





class SimpleContextManager:
    def __init__(self, base_dir: Path | None = None):
        default_dir = Path("service_data") / "sessions"
        self.base_dir = base_dir or default_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, key: SessionKey, ctx: Context) -> Path:
        serializer = JsonSerializer()
        ctx_dict = ctx.to_dict(serializer=serializer)
        path = key.to_path(self.base_dir)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(str(path), 'w', encoding='utf-8') as f:
            json.dump(ctx_dict, f, ensure_ascii=False, indent=4)
        
        return path

    def load(self, key: SessionKey, workflow: Workflow) -> Context:
        path = key.to_path(self.base_dir)
        if not path.exists():
            raise FileNotFoundError(f"No saved Context at {path}")

        with path.open('r', encoding='utf-8') as f:
            ctx_dict = json.load(f)
        
        serializer = JsonSerializer()
        restored = Context.from_dict(workflow=workflow, data=ctx_dict, serializer=serializer)
        return restored
