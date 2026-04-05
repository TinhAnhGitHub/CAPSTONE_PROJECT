import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from rich.syntax import Syntax
from rich.theme import Theme

# ── Theme ─────────────────────────────────────────────────────────────────────

neon_theme = Theme({
    "timestamp":    "dim grey50",
    "agent.header": "bold white on deep_sky_blue1",
    "agent.name":   "bold deep_sky_blue1",
    "user.header":  "bold white on purple3",
    "thought":      "italic orchid1",
    "tool.header":  "bold black on gold1",
    "tool.name":    "bold gold1",
    "success":      "bold spring_green3",
    "error":        "bold red1",
})


class EventHandler:
    def __init__(self):
        self.console = Console(theme=neon_theme, width=120)
        self._streaming = False

    # ── Stream helpers ────────────────────────────────────────────────────────

    def _stream_write(self, text: str) -> None:
        import sys
        sys.stdout.write(text)
        sys.stdout.flush()
        self._streaming = True

    def _stream_flush(self) -> None:
        import sys
        if self._streaming:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._streaming = False

    def _cprint(self, *args, **kwargs) -> None:
        self._stream_flush()
        self.console.print(*args, **kwargs)

    # ── Shared event handlers ─────────────────────────────────────────────────

    def _handle_run_started(self, event: Any) -> None:
        name = getattr(event, "team_name", None) or getattr(event, "agent_name", None) or ""
        model = getattr(event, "model", "") or ""
        provider = getattr(event, "model_provider", "") or ""
        label = "TEAM RUN" if getattr(event, "team_name", None) else "RUN"
        self._cprint(Panel(
            f"[cyan]{label}:[/cyan] [bold]{name}[/bold]\n"
            f"[dim]Model: {provider}/{model}[/dim]",
            title=f"[bold blue]▶ {label} STARTED[/bold blue]",
            border_style="blue",
            box=box.ROUNDED,
        ))

    def _handle_run_content(self, event: Any) -> None:
        chunk = getattr(event, "content", None)
        if chunk:
            self._stream_write(str(chunk))

    def _handle_run_content_completed(self, event: Any) -> None:
        self._stream_flush()

    def _handle_run_completed(self, event: Any) -> None:
        self._stream_flush()
        label = "TEAM RUN" if getattr(event, "team_name", None) else "RUN"
        content = getattr(event, "content", None)
        followups = getattr(event, "followups", None)

        body = Text()
        if content:
            body.append(str(content), style="white")
        if followups:
            body.append("\n\n💡 Followups:\n", style="bold bright_cyan")
            for f in followups:
                body.append(f"  • {f}\n", style="bright_cyan")

        self._cprint(Panel(
            body if body.plain else "[dim]No content[/dim]",
            title=f"[bold green]✔ {label} COMPLETED[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        ))

    def _handle_run_error(self, event: Any) -> None:
        self._stream_flush()
        label = "TEAM RUN" if getattr(event, "team_name", None) else "RUN"
        error_type = getattr(event, "error_type", "") or ""
        content = getattr(event, "content", "") or ""
        self._cprint(Panel(
            f"[red]Type: {error_type}\nMessage: {content}[/red]",
            title=f"[bold red]✘ {label} ERROR[/bold red]",
            border_style="red",
            box=box.ROUNDED,
        ))

    def _handle_run_cancelled(self, event: Any) -> None:
        self._stream_flush()
        reason = getattr(event, "reason", "") or ""
        self._cprint(Panel(
            f"[red]Reason: {reason}[/red]",
            title="[bold red]⊘ RUN CANCELLED[/bold red]",
            border_style="red",
            box=box.ROUNDED,
        ))

    def _handle_run_paused(self, event: Any) -> None:
        self._stream_flush()
        tools = getattr(event, "tools", None) or []
        requirements = getattr(event, "requirements", None) or []
        body = Text()
        for t in tools:
            name = getattr(t, "tool_name", None) or getattr(t, "name", "?")
            body.append(f"  ⚙ Awaiting tool: {name}\n", style="bold orange1")
        for r in requirements:
            body.append(f"  📋 Requirement: {r}\n", style="orange1")
        self._cprint(Panel(
            body if body.plain else "[dim]Paused[/dim]",
            title="[bold orange1]⏸ RUN PAUSED[/bold orange1]",
            border_style="orange1",
            box=box.ROUNDED,
        ))

    def _handle_run_continued(self, event: Any) -> None:
        self._cprint("  ▶  run continued", style="bold orange1")

    def _handle_tool_call_started(self, event: Any) -> None:
        tool = getattr(event, "tool", None)
        tool_name = (getattr(tool, "tool_name", None) or getattr(tool, "name", "?")) if tool else "?"
        tool_id = (getattr(tool, "tool_call_id", "") or "") if tool else ""
        tool_kwargs = (getattr(tool, "tool_args", None) or {}) if tool else {}

        if tool_kwargs:
            table = Table(
                show_header=True,
                header_style="bold yellow",
                border_style="yellow",
                box=box.SIMPLE,
            )
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="white")
            for key, value in tool_kwargs.items():
                table.add_row(str(key), str(value))
            content: Any = table
        else:
            content = "[dim]No parameters[/dim]"

        self._cprint(Panel(
            content,
            title=f"[bold yellow]🔧 Tool Call: {tool_name}[/bold yellow] [dim](ID: {tool_id})[/dim]",
            border_style="yellow",
            box=box.ROUNDED,
        ))

    def _handle_tool_call_completed(self, event: Any) -> None:
        tool = getattr(event, "tool", None)
        tool_name = (getattr(tool, "tool_name", None) or getattr(tool, "name", "?")) if tool else "?"
        result = (getattr(tool, "result", None) if tool else None) or getattr(event, "content", None)
        is_error = getattr(tool, "tool_call_error", False) if tool else False

        if result:
            try:
                parsed = json.loads(result) if isinstance(result, str) else result
                pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
                lang = "json"
            except Exception:
                pretty = str(result)
                lang = "text"
            syntax = Syntax(pretty, lang, theme="monokai", word_wrap=True)
            border = "red" if is_error else "green"
            title_style = "bold red" if is_error else "bold green"
            icon = "✗" if is_error else "✓"
            self._cprint(Panel(
                syntax,
                title=f"[{title_style}]{icon} Tool Result: {tool_name}[/{title_style}]",
                border_style=border,
                box=box.ROUNDED,
            ))
        else:
            self._cprint(
                f"  ✓  tool ← [bold yellow]{tool_name}[/bold yellow]  [dim](no result)[/dim]"
            )

    def _handle_tool_call_error(self, event: Any) -> None:
        tool = getattr(event, "tool", None)
        tool_name = (getattr(tool, "tool_name", None) or getattr(tool, "name", "?")) if tool else "?"
        error = getattr(event, "error", "") or ""
        self._cprint(Panel(
            f"[red]{error}[/red]",
            title=f"[bold red]✘ Tool Error: {tool_name}[/bold red]",
            border_style="red",
            box=box.ROUNDED,
        ))

    def _handle_reasoning_started(self, event: Any) -> None:
        self._cprint("  🧠 reasoning …", style="bold magenta")

    def _handle_reasoning_content_delta(self, event: Any) -> None:
        delta = getattr(event, "reasoning_content", None)
        if delta:
            self._stream_write(delta)

    def _handle_reasoning_step(self, event: Any) -> None:
        content = getattr(event, "reasoning_content", None) or getattr(event, "content", None)
        if content:
            self._cprint(f"  ↳  {content}", style="italic orchid1")

    def _handle_reasoning_completed(self, event: Any) -> None:
        self._stream_flush()
        self._cprint("  🧠 reasoning done", style="bold magenta")

    def _handle_memory_update_started(self, event: Any) -> None:
        self._cprint("  💾 updating memory …", style="bold blue")

    def _handle_memory_update_completed(self, event: Any) -> None:
        memories = getattr(event, "memories", None)
        n = len(memories) if memories else 0
        self._cprint(f"  💾 memory updated ({n} entries)", style="bold blue")

    def _handle_model_request_started(self, event: Any) -> None:
        from rich.markup import escape
        model = escape(getattr(event, "model", "") or "")
        provider = escape(getattr(event, "model_provider", "") or "")
        self._cprint(f"  \u2192 model request  [{provider}/{model}]", style="bold green", markup=False)

    def _handle_model_request_completed(self, event: Any) -> None:
        parts = [
            f"in={getattr(event, 'input_tokens', '-')}",
            f"out={getattr(event, 'output_tokens', '-')}",
        ]
        if getattr(event, "reasoning_tokens", None):
            parts.append(f"reason={event.reasoning_tokens}")
        if getattr(event, "cache_read_tokens", None):
            parts.append(f"cache_r={event.cache_read_tokens}")
        self._cprint(f"  \u2190 model done  [{' '.join(parts)}]", style="bold green", markup=False)

    def _handle_followups_completed(self, event: Any) -> None:
        followups = getattr(event, "followups", None)
        if followups:
            self._cprint("  💡 Suggested followups:", style="bold bright_cyan")
            for f in followups:
                self._cprint(f"     • {f}", style="bright_cyan")

    # ── Team-specific handlers ────────────────────────────────────────────────

    def _handle_task_iteration_started(self, event: Any) -> None:
        iteration = getattr(event, "iteration", 0)
        max_iter = getattr(event, "max_iterations", 0)
        self._cprint(f"  📋 task iteration {iteration + 1}/{max_iter}", style="bold bright_blue")

    def _handle_task_iteration_completed(self, event: Any) -> None:
        iteration = getattr(event, "iteration", 0)
        max_iter = getattr(event, "max_iterations", 0)
        summary = getattr(event, "task_summary", None)
        self._cprint(f"  📋 task iteration {iteration + 1}/{max_iter} completed", style="bold bright_blue")
        if summary:
            self._cprint(f"     Summary: {summary}", style="dim white")

    def _handle_task_state_updated(self, event: Any) -> None:
        tasks = getattr(event, "tasks", None) or []
        goal_complete = getattr(event, "goal_complete", False)
        self._cprint(f"  📋 task state updated ({len(tasks)} tasks)", style="bold bright_blue")
        if goal_complete:
            self._cprint("  ✅ Goal complete!", style="bold green")
        status_icons = {
            "pending": "⏳", "in_progress": "🔄",
            "completed": "✅", "failed": "❌", "blocked": "🚫",
        }
        for task in tasks:
            task_dict = task.to_dict() if hasattr(task, "to_dict") else task
            icon = status_icons.get(task_dict.get("status", ""), "•")
            self._cprint(
                f"     {icon} {task_dict.get('title', 'Untitled')} [{task_dict.get('status', '?')}]",
                style="dim white",
            )

    def _handle_task_created(self, event: Any) -> None:
        title = getattr(event, "title", "Untitled")
        assignee = getattr(event, "assignee", None)
        status = getattr(event, "status", "pending")
        assignee_str = f" → @{assignee}" if assignee else ""
        self._cprint(f"  📋 task created: {title}{assignee_str} [{status}]", style="bold bright_blue")

    def _handle_task_updated(self, event: Any) -> None:
        title = getattr(event, "title", "Untitled")
        status = getattr(event, "status", "")
        prev_status = getattr(event, "previous_status", None)
        status_change = f" {prev_status} → {status}" if prev_status else f" [{status}]"
        self._cprint(f"  📋 task updated: {title}{status_change}", style="bold bright_blue")

    # ── Dispatch tables ───────────────────────────────────────────────────────

    # Keyed by canonical event string (with "Team" prefix already stripped).
    # Both "RunStarted" and "TeamRunStarted" resolve to the same handler.
    _SHARED_HANDLERS: dict[str, str] = {
        "RunStarted":             "_handle_run_started",
        "RunContent":             "_handle_run_content",
        "RunContentCompleted":    "_handle_run_content_completed",
        "RunIntermediateContent": "_handle_run_content",        # same visual treatment
        "RunCompleted":           "_handle_run_completed",
        "RunError":               "_handle_run_error",
        "RunCancelled":           "_handle_run_cancelled",
        "RunPaused":              "_handle_run_paused",
        "RunContinued":           "_handle_run_continued",
        "ToolCallStarted":        "_handle_tool_call_started",
        "ToolCallCompleted":      "_handle_tool_call_completed",
        "ToolCallError":          "_handle_tool_call_error",
        "ReasoningStarted":       "_handle_reasoning_started",
        "ReasoningStep":          "_handle_reasoning_step",
        "ReasoningContentDelta":  "_handle_reasoning_content_delta",
        "ReasoningCompleted":     "_handle_reasoning_completed",
        "MemoryUpdateStarted":    "_handle_memory_update_started",
        "MemoryUpdateCompleted":  "_handle_memory_update_completed",
        "ModelRequestStarted":    "_handle_model_request_started",
        "ModelRequestCompleted":  "_handle_model_request_completed",
        "FollowupsCompleted":     "_handle_followups_completed",
    }

    # Team-only events keyed by their canonical event string.
    _TEAM_ONLY_HANDLERS: dict[str, str] = {
        "TeamTaskIterationStarted":   "_handle_task_iteration_started",
        "TeamTaskIterationCompleted": "_handle_task_iteration_completed",
        "TeamTaskStateUpdated":       "_handle_task_state_updated",
        "TeamTaskCreated":            "_handle_task_created",
        "TeamTaskUpdated":            "_handle_task_updated",
    }

    # ── Reverse-lookup caches (built once, lazily) ────────────────────────────
    _agent_class_to_event: dict[str, str] | None = None
    _team_class_to_event: dict[str, str] | None = None

    def _build_lookup_caches(self) -> None:
        from agno.run.agent import RUN_EVENT_TYPE_REGISTRY
        from agno.run.team import TEAM_RUN_EVENT_TYPE_REGISTRY
        self._agent_class_to_event = {v.__name__: k for k, v in RUN_EVENT_TYPE_REGISTRY.items()}
        self._team_class_to_event = {v.__name__: k for k, v in TEAM_RUN_EVENT_TYPE_REGISTRY.items()}

    # ── Main entry point ──────────────────────────────────────────────────────

    def handle_event(self, raw: dict) -> None:
        """
        Accepts a raw dict from the wire:
          {
            "event_type": "RunStartedEvent",   # dataclass class name
            "agent_id": "...",
            "model": "...",
            ... other event fields ...
          }

        Deserializes it into the appropriate Agno event instance via the
        registry factories, then dispatches to the matching handler method.
        """
        from agno.run.agent import run_output_event_from_dict
        from agno.run.team import team_run_output_event_from_dict

        if self._agent_class_to_event is None:
            self._build_lookup_caches()

        class_name: str = raw.get("event_type", "")
        data = {k: v for k, v in raw.items() if k != "event_type"}

        # Decide agent vs team by presence of team identity fields or Task events
        is_team = bool(data.get("team_id") or data.get("team_name") or "Task" in class_name)

        if is_team:
            event_str = self._team_class_to_event.get(class_name)   # type: ignore[union-attr]
            if event_str is None:
                # Shared dataclass names (e.g. RunStartedEvent) live in agent registry
                event_str = self._agent_class_to_event.get(class_name)  # type: ignore[union-attr]
        else:
            event_str = self._agent_class_to_event.get(class_name)  # type: ignore[union-attr]
            if event_str is None:
                event_str = self._team_class_to_event.get(class_name)  # type: ignore[union-attr]

        if event_str is None:
            return  # unknown event type — silently ignore

        data["event"] = event_str

        # Deserialize into a typed dataclass instance
        if is_team:
            instance = team_run_output_event_from_dict(data)
        else:
            instance = run_output_event_from_dict(data)

        # Dispatch
        ev: str = getattr(instance, "event", "")

        # 1. Team-only task events
        if ev in self._TEAM_ONLY_HANDLERS:
            getattr(self, self._TEAM_ONLY_HANDLERS[ev])(instance)
            return

        # 2. Shared events — strip "Team" prefix so both variants hit the same handler
        lookup = ev[4:] if ev.startswith("Team") else ev
        handler_name = self._SHARED_HANDLERS.get(lookup)
        if handler_name:
            getattr(self, handler_name)(instance)