import json
from datetime import datetime
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.syntax import Syntax
from rich.rule import Rule
from rich.padding import Padding
from rich.style import Style
from rich.theme import Theme

# Llama Index Imports
from llama_index.core.workflow import Event, StopEvent
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentStream,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStreamStructuredOutput
)

# 🎨 BRIGHT NEON THEME FOR DARK TERMINALS
custom_theme = Theme({
    "timestamp": "grey70",
    "agent.name": "bold bright_cyan",
    "agent.label": "bold white on deep_sky_blue1", 
    "tool.name": "bold hot_pink",
    "tool.label": "bold white on deep_pink2",
    "success": "bold spring_green1",
    "warning": "bold gold1",
    "error": "bold red1",
    "header": "bold white",
})

class EventHandler:
    def __init__(self):
        self.console = Console(theme=custom_theme)
        self.last_agent = None
        self.stream_active = False

    def _get_time(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _handle_agent_input(self, data: AgentInput) -> None:
        agent_name = data.current_agent_name
        input_msgs = data.input
        self.last_agent = agent_name

        msg_preview = Text()
        for msg in input_msgs:
            role_style = "bold yellow" if msg.role == "user" else "bold magenta"
            msg_preview.append(f"{msg.role.upper()}: ", style=role_style)
            msg_preview.append(f"{str(msg.content)[:200]}", style="bright_white")
            if len(str(msg.content)) > 200:
                msg_preview.append("...", style="grey50")
            msg_preview.append("\n")

        # Bright Blue Border
        self.console.print(Padding(Panel(
            msg_preview,
            title=f"[agent.label] 🤖 {agent_name} [/agent.label] [grey70]received input[/grey70]",
            border_style="bold dodger_blue1",
            box=box.ROUNDED,
            padding=(0, 2)
        ), (1, 0)))

    def _handle_agent_stream(self, data: AgentStream) -> None:
        if not self.stream_active:
            self.console.print()
            self.console.print(f"[agent.name]⚡ {data.current_agent_name}[/agent.name] [grey70]is working...[/grey70]")
            self.stream_active = True

        delta = data.delta
        thinking_delta = data.thinking_delta

        # Thinking: Lighter purple/grey (readable but distinct)
        if thinking_delta:
            self.console.print(thinking_delta, end='', style="italic plum2")
        # Output: Bright Neon Cyan
        elif delta:
            self.console.print(delta, end='', style="bold cyan1")
        # Heartbeat
        else:
            self.console.print(".", end="", style="dim grey50")

    def _handle_agent_output(self, data: AgentOutput) -> None:
        if self.stream_active:
            self.console.print("\n")
            self.stream_active = False

        agent_name = data.current_agent_name
        response = data.response
        tool_calls = data.tool_calls

        content_group = []
        
        if response.content:
            content_group.append(Text(response.content, style="bright_white"))
        
        if tool_calls:
            t_text = Text("\n🛠️  Triggered Tools:", style="bold gold1")
            for tc in tool_calls:
                t_text.append(f"\n  • {tc.tool_name}", style="khaki1")
            content_group.append(t_text)

        # Bright Green Border
        self.console.print(Padding(Panel(
            Group(*content_group),
            title=f"[agent.label] 💬 {agent_name} [/agent.label] [success]Final Answer[/success]",
            border_style="bold spring_green3",
            box=box.HEAVY_EDGE,
            padding=(1, 2)
        ), (1, 0)))

    def _handle_tool_call(self, data: ToolCall) -> None:
        if self.stream_active:
            self.console.print("\n")
            self.stream_active = False

        tool_name = data.tool_name
        tool_id = data.tool_id
        tool_kwargs = data.tool_kwargs

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold gold1", expand=True)
        table.add_column("Argument", style="bold cyan", ratio=1)
        table.add_column("Value", style="bright_white", ratio=3)

        if tool_kwargs:
            for key, value in tool_kwargs.items():
                table.add_row(key, str(value))
        else:
            table.add_row("-", "[grey50]No parameters[/grey50]")

        # Bright Gold Border
        self.console.print(Padding(Panel(
            table,
            title=f"[tool.label] 🔧 Calling: {tool_name} [/tool.label] [grey70]({tool_id})[/grey70]",
            border_style="bold gold1",
            box=box.ROUNDED,
            padding=(0, 1)
        ), (0, 0, 1, 0)))

    def _handle_tool_call_result(self, data: ToolCallResult):
        tool_name = data.tool_name
        tool_output = data.tool_output
        is_error = tool_output.is_error

        # High Contrast Red vs Bright Green
        style = "bold red1" if is_error else "bold chartreuse3"
        border_style = "red1" if is_error else "chartreuse3"
        icon = "❌" if is_error else "✅"
        
        content_str = str(tool_output)
        try:
            parsed = json.loads(content_str)
            # Monokai is good, but 'ansi_dark' sometimes pops more on black.
            # Keeping monokai but ensuring the surrounding text is bright.
            display_content = Syntax(json.dumps(parsed, indent=2), "json", theme="monokai", word_wrap=True)
        except:
            display_content = Text(content_str, style="bright_white")

        self.console.print(Padding(Panel(
            display_content,
            title=f"[{style}]{icon} Tool Output: {tool_name}[/{style}]",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(1, 2)
        ), (0, 0, 1, 0)))

    def _handle_agent_stream_structured_output(self, data: AgentStreamStructuredOutput) -> None:
        output = data.output
        json_str = json.dumps(output, indent=2)
        syntax = Syntax(json_str, "json", theme="monokai", background_color="default")
        
        # Bright Magenta Border
        self.console.print(Padding(Panel(
            syntax,
            title="[bold magenta1]📊 Structured Output Update[/bold magenta1]",
            border_style="bold magenta1",
            box=box.DOUBLE,
        ), (0, 0)))

    def _handle_stop_event(self, data: StopEvent) -> None:
        if self.stream_active:
            self.console.print("\n")
            self.stream_active = False

        result = data.result
        
        if isinstance(result, (dict, list)):
            content = Syntax(json.dumps(result, indent=2), "json", theme="monokai", line_numbers=True)
        else:
            content = Text(str(result), style="bold bright_white")

        # Neon Green / Success finish
        self.console.print(Padding(Panel(
            content,
            title="[bold white on spring_green3] 🏁 Workflow Completed [/bold white on spring_green3]",
            subtitle=f"[grey70]Finished at {self._get_time()}[/grey70]",
            border_style="bold spring_green3",
            box=box.DOUBLE_EDGE,
            padding=(2, 4)
        ), (2, 0)))

    def handle_event(self, event_data: Event):
        event_type = event_data.__class__.__name__

        handler_map = {
            AgentInput.__name__: self._handle_agent_input,
            AgentOutput.__name__: self._handle_agent_output,
            AgentStream.__name__: self._handle_agent_stream,
            ToolCall.__name__: self._handle_tool_call,
            ToolCallResult.__name__: self._handle_tool_call_result,
            AgentStreamStructuredOutput.__name__: self._handle_agent_stream_structured_output,
            StopEvent.__name__: self._handle_stop_event
        }

        if event_type in handler_map:
            handler_map[event_type](event_data)