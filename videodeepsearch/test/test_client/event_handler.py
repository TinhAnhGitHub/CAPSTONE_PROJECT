import json
from datetime import datetime
from typing import Any
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from rich.syntax import Syntax
from rich.padding import Padding
from rich.align import Align
from rich.theme import Theme
from rich.markdown import Markdown
from rich.layout import Layout
from rich.live import Live

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

# 🎨 THEME DEFINITION
neon_theme = Theme({
    "timestamp": "dim grey50",
    "agent.header": "bold white on deep_sky_blue1",
    "agent.name": "bold deep_sky_blue1",
    "user.header": "bold white on purple3",
    "thought": "italic orchid1",
    "tool.header": "bold black on gold1",
    "tool.name": "bold gold1",
    "success": "bold spring_green3",
    "error": "bold red1",
})

class EventHandler:
    def __init__(self):
        self.console = Console(theme=neon_theme, width=120)
        self.stream_active = False
        self.last_event_type = None

    def _handle_agent_input(self, data_raw: dict) -> None:
        data = AgentInput.model_validate(data_raw)
        agent_name = data.current_agent_name
        input_msgs = data.input
        self.current_agent = agent_name

        self.console.print(Panel(
            f"[cyan]Agent:[/cyan] [bold]{agent_name}[/bold]\n"
            f"[dim]Received {len(input_msgs)} input message(s)[/dim]",
            title="[bold blue]🤖 Agent Input[/bold blue]",
            border_style="blue",
            box=box.ROUNDED
        ))

    def _handle_agent_output(self, data_raw: dict) -> None:
        data = AgentOutput.model_validate(data_raw)
        agent_name = data.current_agent_name
        response = data.response
        content = response.content
        tool_calls = data.tool_calls

        output_text = Text()
        output_text.append(f"Agent: ", style="cyan")
        output_text.append(f"{agent_name}\n", style="bold cyan")

        if content:
            output_text.append(
                f"\n{content}\n", style="white"
            )
        
        if tool_calls:
            output_text.append(f"\n🔧 Tool calls: {len(tool_calls)}", style="yellow dim")
        
        self.console.print(Panel(
            output_text,
            title="[bold green]💬 Agent Response[/bold green]",
            border_style="green",
            box=box.ROUNDED
        ))
        
    def _handle_agent_stream(self, data_raw: dict) -> None:
        """Handle agent stream"""
        data = AgentStream.model_validate(data_raw)
        # if text delta is not there -> thinking
        delta = data.delta
        thinking_delta = data.thinking_delta
        agent_name = data.current_agent_name
        response = data.response

        if thinking_delta:
            self.console.print(
                thinking_delta, end='', style="dim yellow italic"
            )
         
        elif not delta and not thinking_delta:
            self.console.print(".", end="", style="dim yellow")
        
        elif delta:
            self.console.print(delta, end='', style='cyan')
    
    def _handle_tool_call(self, data_raw: dict) -> None:
        data = ToolCall.model_validate(data_raw)
        tool_name = data.tool_name
        tool_id = data.tool_id
        tool_kwargs = data.tool_kwargs

        if tool_kwargs:
            table = Table(show_header=True, header_style="bold yellow", border_style="yellow", box=box.SIMPLE)
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="white")

            for key, value in tool_kwargs.items():
                value_str = str(value)
                table.add_row(key, value_str)
            
            content = table
        else:
            content = "[dim]No parameters[/dim]"
        
        self.console.print(Panel(
            content,
            title=f"[bold yellow]🔧 Tool Call: {tool_name}[/bold yellow] [dim](ID: {tool_id})[/dim]",
            border_style="yellow",
            box=box.ROUNDED
        ))
    

    def _handle_tool_call_result(self, data_raw: dict):
        data = ToolCallResult.model_validate(data_raw)
        tool_name = data.tool_name
        tool_output = data.tool_output

        is_error = tool_output.is_error
        display_content = str(tool_output) # str is the content of the tool output
        if is_error:
            self.console.print(Panel(
                f"[red]{display_content}[/red]",
                title=f"[bold red]✗ Tool Error: {tool_name}[/bold red]",
                border_style="red",
                box=box.ROUNDED
            ))

        self.console.print(Panel(
            f"[green]{display_content}[/green]",
            title=f"[bold green]✓ Tool Result: {tool_name}[/bold green]",
            border_style="green",
            box=box.ROUNDED
        ))



    def _handle_agent_stream_structured_output(self, data_raw: dict) ->None:
        data = AgentStreamStructuredOutput.model_validate(data_raw)
        output = data.output
        try:
            json_str = json.dumps(output, indent=2)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
            
            self.console.print(Panel(
                syntax,
                title="[bold magenta]📊 Structured Output Stream[/bold magenta]",
                border_style="magenta",
                box=box.ROUNDED
            ))
        except Exception as e:
            self.console.print(Panel(
                f"[yellow]{str(output)}[/yellow]",
                title="[bold magenta]📊 Structured Output Stream[/bold magenta]",
                border_style="magenta",
                box=box.ROUNDED
            ))

    def _handle_stop_event(self, data_raw: dict) -> None:
        """The final result of the workflow."""
        ...

    def handle_event(self, event_data: dict):
        event_type = event_data['event_type']
        
        handlers = {
            AgentInput.__name__: self._handle_agent_input,
            AgentOutput.__name__: self._handle_agent_output,
            AgentStream.__name__: self._handle_agent_stream,
            ToolCall.__name__: self._handle_tool_call,
            ToolCallResult.__name__: self._handle_tool_call_result,
            StopEvent.__name__: self._handle_stop_event
        }

        if event_type in handlers:
            handlers[event_type](event_data)