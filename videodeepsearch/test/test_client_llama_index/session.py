from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel

from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich import box

from llama_index.core.llms import ChatMessage

class Session:
    def __init__(self, user_id: str, session_id: str, chat_history: list[ChatMessage] | None = None):
        self.user_id = user_id
        self.session_id = session_id
        self.chat_history = chat_history or []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
    
    def add_message(self, message: list[ChatMessage]):
        self.chat_history.extend(message)
    
    def clear_history(self):
        """Clear all chat history."""
        self.chat_history = []
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "chat_history": [chat.model_dump() for chat  in self.chat_history],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            'session_id': self.session_id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        chat_history_raw = data.get('chat_history', [])

        chat_histories = [ChatMessage.model_validate(chat) for chat in chat_history_raw]
        
        session = cls(
            user_id=data["user_id"],
            session_id=data['session_id'],
            chat_history=chat_histories
        )
        session.created_at = data.get("created_at", session.created_at)
        session.updated_at = data.get("updated_at", session.updated_at)
        return session
    
    def display_history(self, console: Console):
        if not self.chat_history:
            console.print(Panel(
                "[dim]No messages in history[/dim]",
                title="[bold yellow]Chat History[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED
            ))
            return
        
        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="blue",
            box=box.ROUNDED,
            title=f"[bold]Chat History[/bold] [dim]({len(self.chat_history)} messages)[/dim]",
            title_style="bold yellow"
        )
        
        table.add_column("#", style="dim", width=4)
        table.add_column("Role", style="cyan", width=10)
        table.add_column("Content", style="white")
        
        for i, msg in enumerate(self.chat_history, 1):
            role = msg.role
            content = msg.content
            
            role_style = "green" if role == "user" else "blue"
            icon = "👤" if role == "user" else "🤖"
            
            display_content = content[:150] + "..." if len(content) > 150 else content #type:ignore
            
            table.add_row(
                str(i),
                f"{icon} {role.upper()}",
                display_content,
                style=role_style if i % 2 == 0 else None
            )
        
        console.print()
        console.print(table)
        console.print()


class SessionManager:
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.session_dir.mkdir(exist_ok=True)
    
    def _get_session_path(self, user_id: str, session_id: str) -> Path:
        """Get the file path for a user's session."""
        safe_user_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
        return self.session_dir / f"{safe_user_id}" / f"{session_id}.json"

    def load_session(self, user_id: str, session_id: str) -> Session:
        session_path = self._get_session_path(user_id, session_id)
        session_path.parent.mkdir(parents=True, exist_ok=True)

        if session_path.exists():
            with open(session_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Session.from_dict(data)
        
        return Session(user_id, session_id)

    def save_session(self, session: Session):
        """Save session to file."""
        session_path = self._get_session_path(session.user_id, session.session_id)
        
        try:
            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            try:
                from rich.console import Console
                Console().print(f"[yellow]⚠ Error saving session:[/yellow] {e}")
            except ImportError:
                print(f"⚠️ Error saving session: {e}")

    def delete_session(self, user_id: str, session_id:str):
        """Delete a session file."""
        session_path = self._get_session_path(user_id, session_id)
        if session_path.exists():
            session_path.unlink()
    
    def list_sessions(self) -> list[str]:
        """List all available session user IDs."""
        return [
            f.stem for f in self.session_dir.glob("*.json")
        ]
    
