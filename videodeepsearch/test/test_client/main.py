import asyncio
import logging
import sys
from pathlib import Path
import uuid
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich import box
from llama_index.core.llms import ChatMessage, MessageRole

from client import WorkflowClient
from session import SessionManager
from event_handler import EventHandler

from llama_index.core.evaluation import RelevancyEvaluator, EvaluationResult
from llama_index.llms.gemini import Gemini


import json

async def run_interactive_session(test_session_dir: Path, user_id: str, list_video_ids:list[str], mode = 1):
    for name in ("websockets", "websockets.client", "websockets.server", "websockets.protocol"):
        logging.getLogger(name).setLevel(logging.WARNING)

    console = Console()
    session_dir = test_session_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)

    session_manager = SessionManager(session_dir)
    event_handler = EventHandler(console=console)
    if mode == 1:
        websocket_url = "ws://100.113.186.28:8050/ws/start_workflow"
    else:
        websocket_url = "ws://localhost:8050/ws/start_workflow"

    console.print(Panel.fit(
        "[bold cyan]🎬 Video Deep Search Workflow Client[/bold cyan]\n"
        "[dim]Interactive CLI for testing workflow services[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))

    console.print()

    session_id = Prompt.ask("[bold magenta]Enter session_id (Press Enter to create a new one): [/bold magenta]").strip()
    
    from datetime import datetime
    if not session_id:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        console.print(f"[green]✓[/green] Created new session with ID: [cyan]{session_id}[/cyan]")


    console.print("\n[bold yellow]Available Commands:[/bold yellow]")
    console.print("  [cyan]•[/cyan] Type your question to start workflow")
    console.print("  [cyan]•[/cyan] [bold]history[/bold] - View chat history")
    console.print("  [cyan]•[/cyan] [bold]clear[/bold] - Clear session")
    console.print("  [cyan]•[/cyan] [bold]exit[/bold] or [bold]quit[/bold] - Quit\n")
    
    client = WorkflowClient(
        websocket_url=websocket_url,
        user_id=user_id,
        event_handler=event_handler
    )
    
    while True:
        session = session_manager.load_session(user_id, session_id)

        with open("test.json", 'r') as f:
            res = json.load(f)
        try:
            console.print()
            console.print(Panel(
                f"[green]✓[/green] Session loaded: [bold]{len(session.chat_history)}[/bold] messages in history\n"
                f"[dim]User:[/dim] [cyan]{user_id}[/cyan]\n"
                f"[dim]Last updated:[/dim] {session.updated_at}",
                title="[bold]Session Info[/bold]",
                border_style="green",
                box=box.ROUNDED
            ))

            user_input = Prompt.ask("[bold magenta]You[/bold magenta]").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit']:
                console.print("\n[bold green]👋 Goodbye![/bold green]")
                break
            
            if user_input.lower() == 'history':
                session.display_history(console)
                continue
            
            if user_input.lower() == 'clear':
                session.clear_history()
                session_manager.save_session(session)
                console.print("[green]✓[/green] Session cleared", style="bold")
                continue
            
            
            console.print()
            console.rule("[bold cyan]Starting Workflow[/bold cyan]", style="cyan")
            console.print()
            
            final_chat_message = await client.execute_workflow(
                user_demand=user_input,
                video_ids=list_video_ids,
                chat_history=session.chat_history,
                session_id=session_id
            )
            
            if final_chat_message:
                final_message = final_chat_message
                session.add_message(final_message)
                session_manager.save_session(session)
                
                console.print()
                console.print(Panel(
                    "[green]✓[/green] Workflow complete. Session saved.",
                    border_style="green",
                    box=box.ROUNDED
                ))
                try:
                    evaluator_llm = Gemini(temperature = 0.0)
                    relevancy_evaluator = RelevancyEvaluator(llm=evaluator_llm)
                    response = final_chat_message[-1].content
                    eval_result: EvaluationResult = await relevancy_evaluator.aevaluate_response(
                        query=user_input, 
                        response=response,    
                    )
                    res.append({
                        "question": user_input,
                        "result": res,
                        "evaluation":{
                            "passing":eval_result.passing,
                            "score": eval_result.score,
                            "feedback": eval_result.feedback
                        }
                    })
                except Exception as e:
                    res.append({"error": e})
                with open("test.json", "w", encoding="utf-8") as f:
                    json.dump(res, f, ensure_ascii=False, indent=4)
            else:
                console.print("\n[yellow]⚠[/yellow] Workflow completed without final response", style="dim")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]⚠ Interrupted by user[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


if __name__ == "__main__":
    test_session_dir = Path('../local')
    user_id = 'testagent'
    list_video_ids = []
    mode = 2
    try:
        asyncio.run(run_interactive_session(test_session_dir=test_session_dir, user_id=user_id, list_video_ids=list_video_ids, mode = mode))
    except KeyboardInterrupt:
        console = Console()
        console.print("\n[bold green]👋 Goodbye![/bold green]")
        sys.exit(0)
    
