"""
print_run_event.py
------------------
Pretty-print any RunOutputEvent or TeamRunOutputEvent to the CLI using `rich`.

Install dependency:
    pip install rich
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.markup import escape
from rich.style import Style
from rich.text import Text

if TYPE_CHECKING:
    from agno.run.response import RunOutputEvent  # type: ignore
    from agno.run.team import TeamRunOutputEvent  # type: ignore

console = Console()

# ── Streaming state ───────────────────────────────────────────────────────────
_streaming = False


def _stream_write(text: str) -> None:
    """Write a raw chunk directly to stdout — no rich processing, no auto-newline."""
    global _streaming
    sys.stdout.write(text)
    sys.stdout.flush()
    _streaming = True


def _stream_flush() -> None:
    """Close an open stream line. Call before any rich output after streaming."""
    global _streaming
    if _streaming:
        sys.stdout.write("\n")
        sys.stdout.flush()
        _streaming = False


# ── Palette ───────────────────────────────────────────────────────────────────
_C = {
    "run":       "bold cyan",
    "tool":      "bold yellow",
    "reasoning": "bold magenta",
    "memory":    "bold blue",
    "model":     "bold green",
    "hook":      "dim white",
    "compress":  "bold dark_orange",
    "followup":  "bold bright_cyan",
    "error":     "bold red",
    "cancel":    "bold red",
    "pause":     "bold orange1",
    "content":   "white",
    "dim":       "dim white",
    "agent":     "bright_black",
    "team":      "bold bright_magenta",
    "task":      "bold bright_blue",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _meta(event: Any) -> str:
    """Extract metadata (agent/team name, run_id) from an event."""
    parts = []
    name = (
        getattr(event, "team_name", None)
        or getattr(event, "team_id", None)
        or getattr(event, "agent_name", None)
        or getattr(event, "agent_id", None)
    )
    if name:
        parts.append(name)
    run_id = getattr(event, "run_id", None)
    if run_id:
        parts.append(f"run={run_id}")
    return "  " + " · ".join(parts) if parts else ""


def _header(label: str, color: str, event: Any) -> None:
    """Flush any open stream, then print a rule header."""
    _stream_flush()
    meta = _meta(event)
    console.rule(
        Text.assemble(
            (f" {label} ", Style.parse(color)),
            (meta, Style.parse(_C["agent"])),
        ),
        style=color,
        align="left",
    )


def _kv(key: str, value: Any, indent: int = 2) -> None:
    if value is None or value == "" or value == []:
        return
    pad = " " * indent
    console.print(
        Text.assemble(
            (f"{pad}{key}: ", Style.parse("bold bright_black")),
            (escape(str(value)), Style.parse(_C["content"])),
        )
    )


def _cprint(*args, **kwargs) -> None:
    """Flush stream state before any regular rich console.print call."""
    _stream_flush()
    console.print(*args, **kwargs)


def _tool_summary(tool: Any) -> str:
    name = getattr(tool, "tool_name", None) or getattr(tool, "name", "?")
    tid  = getattr(tool, "tool_call_id", "") or ""
    return f"{name}({tid})" if tid else name


def _fmt_args(tool: Any) -> str:
    """Format tool_args dict as key=value pairs — full output, no truncation."""
    import json
    args = getattr(tool, "tool_args", None)
    if not args:
        return ""
    try:
        rendered = "  ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
    except Exception:
        rendered = str(args)
    return rendered


def _is_team_event(ev: str) -> bool:
    """Check if an event is a Team event (prefixed with 'Team')."""
    return ev.startswith("Team")


def _strip_team_prefix(ev: str) -> str:
    """Remove 'Team' prefix from event name for display."""
    return ev[4:] if ev.startswith("Team") else ev


# ── Main dispatcher ───────────────────────────────────────────────────────────

def print_run_event(event: Any, *, verbose: bool = True, show_tool_results: bool = True) -> None:
    """
    Print a single RunOutputEvent or TeamRunOutputEvent to the terminal cleanly.

    Parameters
    ----------
    event:
        Any event from `agno.run.response.RunOutputEvent` or `agno.run.team.TeamRunOutputEvent`.
    verbose:
        If True, print extra fields like token counts and step IDs.
    show_tool_results:
        If True, pretty-print the full tool result below each ToolCallCompleted line.
    """
    ev = getattr(event, "event", None) or type(event).__name__
    is_team = _is_team_event(ev)
    display_ev = _strip_team_prefix(ev) if is_team else ev
    event_color = _C["team"] if is_team else _C["run"]

    # ── Run lifecycle ─────────────────────────────────────────────────────────
    if ev in ("RunStarted", "TeamRunStarted"):
        label = "▶  TEAM RUN STARTED" if is_team else "▶  RUN STARTED"
        color = _C["team"] if is_team else _C["run"]
        _header(label, color, event)
        model    = getattr(event, "model", None)
        provider = getattr(event, "model_provider", None)
        if model or provider:
            _kv("model", f"{provider}/{model}")

    elif ev in ("RunContent", "TeamRunContent"):
        chunk = getattr(event, "content", None)
        if chunk:
            _stream_write(str(chunk))

    elif ev in ("RunIntermediateContent", "TeamRunIntermediateContent"):
        chunk = getattr(event, "content", None)
        if chunk:
            _stream_write(str(chunk))

    elif ev in ("RunContentCompleted", "TeamRunContentCompleted"):
        _stream_flush()

    elif ev in ("RunCompleted", "TeamRunCompleted"):
        _stream_flush()
        label = "✔  TEAM RUN COMPLETED" if is_team else "✔  RUN COMPLETED"
        color = _C["team"] if is_team else _C["run"]
        _header(label, color, event)
        followups = getattr(event, "followups", None)
        if followups:
            _cprint("  💡 Followups:", style=_C["followup"])
            for f in followups:
                _cprint(f"     • {f}", style=_C["followup"])
        # Show member responses for team runs
        member_responses = getattr(event, "member_responses", None)
        if member_responses and is_team:
            _cprint(f"  📦 Member responses: {len(member_responses)}", style=_C["team"])
        if verbose:
            metrics = getattr(event, "metrics", None)
            if metrics:
                _kv("tokens", f"in={getattr(metrics,'input_tokens','-')} out={getattr(metrics,'output_tokens','-')}")

    elif ev in ("RunError", "TeamRunError"):
        label = "✘  TEAM RUN ERROR" if is_team else "✘  RUN ERROR"
        color = _C["team"] if is_team else _C["error"]
        _header(label, color, event)
        _kv("type",    getattr(event, "error_type", None))
        _kv("message", getattr(event, "content", None))
        if verbose:
            extra = getattr(event, "additional_data", None)
            if extra:
                _kv("data", extra)

    elif ev in ("RunCancelled", "TeamRunCancelled"):
        label = "⊘  TEAM RUN CANCELLED" if is_team else "⊘  RUN CANCELLED"
        color = _C["team"] if is_team else _C["cancel"]
        _header(label, color, event)
        _kv("reason", getattr(event, "reason", None))

    elif ev in ("RunPaused", "TeamRunPaused"):
        label = "⏸  TEAM RUN PAUSED" if is_team else "⏸  RUN PAUSED"
        color = _C["team"] if is_team else _C["pause"]
        _header(label, color, event)
        for t in (getattr(event, "tools", None) or []):
            _kv("awaiting tool", _tool_summary(t))
        for r in (getattr(event, "requirements", None) or []):
            _kv("requirement", str(r))

    elif ev in ("RunContinued", "TeamRunContinued"):
        label = "TEAM run continued" if is_team else "run continued"
        _cprint(f"  ▶  {label}", style=_C["team"] if is_team else _C["pause"])

    # ── Tool calls ────────────────────────────────────────────────────────────
    elif ev in ("ToolCallStarted", "TeamToolCallStarted"):
        prefix = "TEAM " if is_team else ""
        t = getattr(event, "tool", None)
        name = _tool_summary(t) if t else "?"
        args_str = _fmt_args(t) if t else ""
        line = Text.assemble(
            (f"  ⚙  {prefix}tool → ", Style.parse(_C["tool"])),
            (name,           Style.parse("bold yellow")),
        )
        if args_str:
            line.append(f"\n     args: {args_str}", style="dim yellow")
        _cprint(line)

    elif ev in ("ToolCallCompleted", "TeamToolCallCompleted"):
        import json
        prefix = "TEAM " if is_team else ""
        t = getattr(event, "tool", None)
        name = _tool_summary(t) if t else "?"
        args_str = _fmt_args(t) if t else ""

        # Prefer tool.result (ToolExecution.result), fall back to event.content
        result = (getattr(t, "result", None) if t else None) or getattr(event, "content", None)

        # Surface tool_call_error flag if set
        tool_call_error = getattr(t, "tool_call_error", None) if t else None

        line = Text.assemble(
            (f"  ✓  {prefix}tool ← ", Style.parse(_C["tool"])),
            (name,           Style.parse("bold yellow")),
        )
        if tool_call_error:
            line.append("  [ERROR]", style=_C["error"])
        if args_str:
            line.append(f"\n     args: {args_str}", style="dim yellow")
        _cprint(line)

        if result:
            try:
                parsed = json.loads(result) if isinstance(result, str) else result
                pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
                lang = "json"
            except Exception:
                pretty = str(result)
                lang = "text"
            from rich.syntax import Syntax
            from rich.panel import Panel
            syntax = Syntax(pretty, lang, theme="ansi_dark", word_wrap=True)
            border = "dim red" if tool_call_error else "dim yellow"
            _cprint(Panel(syntax, title=f"[bold yellow]{name}[/] result",
                          border_style=border, padding=(0, 1)))

        if verbose:
            images = getattr(event, "images", None)
            videos = getattr(event, "videos", None)
            audio  = getattr(event, "audio", None)
            if images: _kv("images", len(images))
            if videos: _kv("videos", len(videos))
            if audio:  _kv("audio",  len(audio))

    elif ev in ("ToolCallError", "TeamToolCallError"):
        prefix = "TEAM " if is_team else ""
        t = getattr(event, "tool", None)
        name = _tool_summary(t) if t else "?"
        args_str = _fmt_args(t) if t else ""
        err = getattr(event, "error", "") or ""
        line = Text.assemble(
            (f"  ✘  {prefix}tool error ← ", Style.parse(_C["error"])),
            (name,                 Style.parse("bold red")),
        )
        if args_str:
            line.append(f"\n     args: {args_str}", style="dim red")
        if err:
            line.append(f"\n     error: {err}", style=_C["error"])
        _cprint(line)

    # ── Reasoning ─────────────────────────────────────────────────────────────
    elif ev in ("ReasoningStarted", "TeamReasoningStarted"):
        prefix = "TEAM " if is_team else ""
        _cprint(f"  🧠 {prefix}reasoning …", style=_C["reasoning"])

    elif ev in ("ReasoningContentDelta", "TeamReasoningContentDelta"):
        delta = getattr(event, "reasoning_content", None)
        if delta:
            _stream_write(delta)

    elif ev in ("ReasoningStep", "TeamReasoningStep"):
        content = getattr(event, "reasoning_content", None) or getattr(event, "content", None)
        if content:
            _cprint(f"  ↳  {escape(str(content))}", style=_C["reasoning"])

    elif ev in ("ReasoningCompleted", "TeamReasoningCompleted"):
        _stream_flush()
        prefix = "TEAM " if is_team else ""
        _cprint(f"  🧠 {prefix}reasoning done", style=_C["reasoning"])

    # ── Memory ────────────────────────────────────────────────────────────────
    elif ev in ("MemoryUpdateStarted", "TeamMemoryUpdateStarted"):
        prefix = "TEAM " if is_team else ""
        _cprint(f"  💾 {prefix}updating memory …", style=_C["memory"])

    elif ev in ("MemoryUpdateCompleted", "TeamMemoryUpdateCompleted"):
        prefix = "TEAM " if is_team else ""
        memories = getattr(event, "memories", None)
        n = len(memories) if memories else 0
        _cprint(f"  💾 {prefix}memory updated ({n} entries)", style=_C["memory"])

    # ── Session summary ───────────────────────────────────────────────────────
    elif ev in ("SessionSummaryStarted", "TeamSessionSummaryStarted"):
        prefix = "TEAM " if is_team else ""
        _cprint(f"  📝 {prefix}summarising session …", style=_C["memory"])

    elif ev in ("SessionSummaryCompleted", "TeamSessionSummaryCompleted"):
        prefix = "TEAM " if is_team else ""
        _cprint(f"  📝 {prefix}session summary ready", style=_C["memory"])
        if verbose:
            summary = getattr(event, "session_summary", None)
            if summary:
                _kv("summary", str(summary))

    # ── Model requests ────────────────────────────────────────────────────────
    elif ev in ("ModelRequestStarted", "TeamModelRequestStarted"):
        if verbose:
            prefix = "TEAM " if is_team else ""
            model    = getattr(event, "model", None)
            provider = getattr(event, "model_provider", None)
            _cprint(f"  → {prefix}model request  [{provider}/{model}]", style=_C["model"])

    elif ev in ("ModelRequestCompleted", "TeamModelRequestCompleted"):
        if verbose:
            prefix = "TEAM " if is_team else ""
            parts = [
                f"in={getattr(event, 'input_tokens', '-')}",
                f"out={getattr(event, 'output_tokens', '-')}",
            ]
            if getattr(event, "reasoning_tokens", None):
                parts.append(f"reason={event.reasoning_tokens}")
            if getattr(event, "cache_read_tokens", None):
                parts.append(f"cache_r={event.cache_read_tokens}")
            _cprint(f"  ← {prefix}model done  [{' '.join(parts)}]", style=_C["model"])

    # ── Parser / output model ─────────────────────────────────────────────────
    elif ev in ("ParserModelResponseStarted", "TeamParserModelResponseStarted"):
        if verbose:
            prefix = "TEAM " if is_team else ""
            _cprint(f"  ⟳  {prefix}parser …", style=_C["dim"])

    elif ev in ("ParserModelResponseCompleted", "TeamParserModelResponseCompleted"):
        if verbose:
            prefix = "TEAM " if is_team else ""
            _cprint(f"  ✓  {prefix}parser done", style=_C["dim"])

    elif ev in ("OutputModelResponseStarted", "TeamOutputModelResponseStarted"):
        if verbose:
            prefix = "TEAM " if is_team else ""
            _cprint(f"  ⟳  {prefix}output model …", style=_C["dim"])

    elif ev in ("OutputModelResponseCompleted", "TeamOutputModelResponseCompleted"):
        if verbose:
            prefix = "TEAM " if is_team else ""
            _cprint(f"  ✓  {prefix}output model done", style=_C["dim"])

    # ── Hooks ─────────────────────────────────────────────────────────────────
    elif ev in ("PreHookStarted", "TeamPreHookStarted"):
        if verbose:
            prefix = "TEAM " if is_team else ""
            _cprint(f"  ↪ {prefix}pre-hook: {getattr(event, 'pre_hook_name', '')}", style=_C["hook"])

    elif ev in ("PreHookCompleted", "TeamPreHookCompleted"):
        if verbose:
            prefix = "TEAM " if is_team else ""
            _cprint(f"  ↩ {prefix}pre-hook done: {getattr(event, 'pre_hook_name', '')}", style=_C["hook"])

    elif ev in ("PostHookStarted", "TeamPostHookStarted"):
        if verbose:
            prefix = "TEAM " if is_team else ""
            _cprint(f"  ↪ {prefix}post-hook: {getattr(event, 'post_hook_name', '')}", style=_C["hook"])

    elif ev in ("PostHookCompleted", "TeamPostHookCompleted"):
        if verbose:
            prefix = "TEAM " if is_team else ""
            _cprint(f"  ↩ {prefix}post-hook done: {getattr(event, 'post_hook_name', '')}", style=_C["hook"])

    # ── Compression ───────────────────────────────────────────────────────────
    elif ev in ("CompressionStarted", "TeamCompressionStarted"):
        prefix = "TEAM " if is_team else ""
        _cprint(f"  🗜  {prefix}compressing tool results …", style=_C["compress"])

    elif ev in ("CompressionCompleted", "TeamCompressionCompleted"):
        prefix = "TEAM " if is_team else ""
        orig = getattr(event, "original_size", None) or 0
        comp = getattr(event, "compressed_size", None) or 0
        pct  = round((1 - comp / orig) * 100) if orig else 0
        n    = getattr(event, "tool_results_compressed", "?")
        _cprint(
            f"  🗜  {prefix}compressed {n} result(s)  {orig:,} → {comp:,} chars  (↓{pct}%)",
            style=_C["compress"],
        )

    # ── Followups ─────────────────────────────────────────────────────────────
    elif ev in ("FollowupsStarted", "TeamFollowupsStarted"):
        if verbose:
            prefix = "TEAM " if is_team else ""
            _cprint(f"  💡 {prefix}generating followups …", style=_C["followup"])

    elif ev in ("FollowupsCompleted", "TeamFollowupsCompleted"):
        prefix = "TEAM " if is_team else ""
        followups = getattr(event, "followups", None)
        if followups:
            _cprint(f"  💡 {prefix}Suggested followups:", style=_C["followup"])
            for f in followups:
                _cprint(f"     • {f}", style=_C["followup"])

    # ── Task events (Team only) ───────────────────────────────────────────────
    elif ev == "TeamTaskIterationStarted":
        iteration = getattr(event, "iteration", 0)
        max_iter  = getattr(event, "max_iterations", 0)
        _cprint(
            f"  📋 task iteration {iteration + 1}/{max_iter}",
            style=_C["task"],
        )

    elif ev == "TeamTaskIterationCompleted":
        iteration = getattr(event, "iteration", 0)
        max_iter  = getattr(event, "max_iterations", 0)
        summary   = getattr(event, "task_summary", None)
        _cprint(
            f"  📋 task iteration {iteration + 1}/{max_iter} completed",
            style=_C["task"],
        )
        if summary and verbose:
            _kv("summary", summary)

    elif ev == "TeamTaskStateUpdated":
        tasks = getattr(event, "tasks", None) or []
        goal_complete = getattr(event, "goal_complete", False)
        completion_summary = getattr(event, "completion_summary", None)
        _cprint(
            f"  📋 task state updated ({len(tasks)} tasks)",
            style=_C["task"],
        )
        if goal_complete:
            _cprint("  ✅ Goal complete!", style="bold green")
        if completion_summary and verbose:
            _kv("completion", completion_summary)
        if verbose and tasks:
            for task in tasks:
                task_dict = task.to_dict() if hasattr(task, "to_dict") else task
                status_icon = {"pending": "⏳", "in_progress": "🔄", "completed": "✅", "failed": "❌", "blocked": "🚫"}.get(task_dict.get("status", ""), "•")
                _cprint(f"     {status_icon} {task_dict.get('title', 'Untitled')} [{task_dict.get('status', '?')}]", style=_C["dim"])

    elif ev == "TeamTaskCreated":
        task_id = getattr(event, "task_id", "")
        title = getattr(event, "title", "Untitled")
        assignee = getattr(event, "assignee", None)
        status = getattr(event, "status", "pending")
        assignee_str = f" → @{assignee}" if assignee else ""
        _cprint(
            f"  📋 task created: {title}{assignee_str} [{status}]",
            style=_C["task"],
        )

    elif ev == "TeamTaskUpdated":
        task_id = getattr(event, "task_id", "")
        title = getattr(event, "title", "Untitled")
        status = getattr(event, "status", "")
        prev_status = getattr(event, "previous_status", None)
        result = getattr(event, "result", None)
        status_change = f" {prev_status} → {status}" if prev_status else f" [{status}]"
        _cprint(
            f"  📋 task updated: {title}{status_change}",
            style=_C["task"],
        )
        if result and verbose:
            _kv("result", result)

    # ── Custom ────────────────────────────────────────────────────────────────
    elif ev == "CustomEvent":
        if verbose:
            prefix = "TEAM " if is_team else ""
            _cprint(f"  ◈  {prefix}custom event", style=_C["dim"])
            for k, v in vars(event).items():
                if k not in {"event", "created_at", "team_id", "team_name", "agent_id", "agent_name",
                             "run_id", "session_id"}:
                    _kv(k, v)

    else:
        if verbose:
            prefix = "TEAM " if is_team else ""
            _cprint(f"  ?  {prefix}{display_ev}", style=_C["dim"])


# ── Convenience: print a whole RunOutput ─────────────────────────────────────

def print_run_output(run_output: Any, *, verbose: bool = False, show_tool_results: bool = False) -> None:
    """
    Iterate over all events in a RunOutput or TeamRunOutput and print each one.

    Usage:
        from print_run_event import print_run_output
        print_run_output(my_run_output, show_tool_results=True)
    """
    events = getattr(run_output, "events", None) or []
    for event in events:
        print_run_event(event, verbose=verbose, show_tool_results=show_tool_results)
    _stream_flush()
    console.print()