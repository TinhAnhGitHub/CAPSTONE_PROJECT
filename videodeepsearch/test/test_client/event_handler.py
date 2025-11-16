from typing import Any
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box    
from rich.syntax import Syntax
import json
from pydantic import BaseModel

from llama_index.core.agent.workflow import (
    AgentInput,
    AgentSetup,
    AgentStream,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStreamStructuredOutput
)
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from videodeepsearch.agent.orc_events import (
    UserInputEvent,
    FinalResponseEvent,
    PlannerInputEvent,
    PlanProposedEvent,
    AgentProgressEvent,
    AgentDecision,
    PlanningAgentEvent,
    FinalEvent,
    StopEvent,
    EvaluationCompleteEvent
)


class EventHandler:
    def __init__(self, console: Console):
        self.console = console
        self.event_count = 0
        self.current_agent = None
    
    def _handle_agent_input(self, data: AgentInput) -> None:
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

    def _handle_agent_output(self, data: AgentOutput) -> None:
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
        
    def _handle_agent_stream(self, data: AgentStream) -> None:
        """Handle agent stream"""

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
    
    def _handle_tool_call(self, data: ToolCall) -> None:
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
    

    def _handle_tool_call_result(self, data: ToolCallResult):
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

    def _handle_user_input(self, data: UserInputEvent) -> None:
        user_demand  = data.user_demand
        video_ids = data.list_video_ids

        output = Text()
        output.append("Request: ", style="bold magenta")
        output.append(f"{user_demand}\n", style="white")

        if video_ids:
            output.append("\n🎥 Video IDs: ", style="cyan")
            output.append(", ".join(video_ids), style="cyan dim")
        
        self.console.print(Panel(
            output,
            title="[bold magenta]📥 User Input[/bold magenta]",
            border_style="magenta",
            box=box.DOUBLE
        ))
    
    def _handle_final_response(self, data: FinalResponseEvent) -> None:
        messages = data.passing_messages
        message_str = []
        for mess in messages:
            for block in mess.blocks:
                message_str.append(str(block))

        message_full = '\n'.join(message_str)

        self.console.print(Panel(
            f"[green]✓[/green] Preparing final response\n"
            f"[dim]Processing {len(messages)} message(s)[/dim]",
            title="[bold green]📤 Final Response Event[/bold green]",
            border_style="green",
            box=box.ROUNDED
        ))

        self.console.print(Panel(
            f"[white]{message_full}[/white]",
            title="[bold green]💬 Final Answer[/bold green]",
            border_style="bright_green",
            box=box.ROUNDED
        ))

    def _handle_planner_input(self, data: PlannerInputEvent) -> None:
        user_msg = data.user_msg
        planner_demand = data.planner_demand

        output = Text()
        output.append("User Message:\n", style="bold yellow")
        output.append(f"{user_msg}\n\n", style="white")
        output.append("Planning Demand:\n", style="bold yellow")
        output.append(f"{planner_demand}", style="white")
        
        self.console.print(Panel(
            output,
            title="[bold yellow]📋 Planner Input[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED
        ))

    
    def _handle_plan_proposed(self, data: PlanProposedEvent): 
        agent_response = data.agent_response
        worker_plan = data.worker_plan

        output = Text()
        output.append("Planner Response:\n", style="bold cyan")
        output.append(f"{agent_response[:200]}{'...' if len(agent_response) > 200 else ''}\n", style="white")
        
        if hasattr(worker_plan, "plan_detail") and worker_plan.plan_detail:
            workers_blueprint = worker_plan.plan_detail
            output.append("\n👷 Worker Blueprints:\n", style="bold cyan")

            for idx, blueprint in enumerate(workers_blueprint, start=1):
                name = getattr(blueprint, "name", blueprint.name)
                description = getattr(blueprint, "description", blueprint.description)
                task = getattr(blueprint, "task", blueprint.task)
                tools = getattr(blueprint, "tools", blueprint.tools)

                output.append(f"\n[{idx}] [bold]{name}[/bold]\n", style="cyan")
                output.append(f"• [bold]Description:[/bold] {description}\n", style="white")
                output.append(f"• [bold]Task:[/bold] {task}\n", style="white")
                output.append(f"• [bold]Tools:[/bold] {', '.join(tools) if tools else 'None'}\n", style="white")

        else:
            output.append("\n[dim]No worker blueprint found in plan_detail.[/dim]\n")

        self.console.print(Panel(
            output,
            title="[bold cyan]📝 Plan Proposed[/bold cyan]",
            border_style="cyan",
            box=box.HEAVY
        ))

    def _handle_agent_progress(self, data: AgentProgressEvent) -> None:
        """Handle AgentProgressEvent."""
        agent_name = data.agent_name
        answer = data.answer

        answer_str = str(answer)
        display_answer = answer_str[:150] + "..." if len(answer_str) > 150 else answer_str
        
        self.console.print(Panel(
            f"[bold blue]{agent_name}[/bold blue]\n[dim]{display_answer}[/dim]",
            title="[bold blue]⏳ Agent Progress[/bold blue]",
            border_style="blue",
            box=box.ROUNDED
        ))
    
    def _handle_planning_agent(self, data: PlanningAgentEvent) -> None:
        reason = data.reason
        plan_summary = data.plan_summary
        plan_detail = data.plan_detail

        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            box=box.ROUNDED,
            title="Plan Details",
            title_style="bold"
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Step", style="white")
        
        for i, detail in enumerate(plan_detail[:10], 1):  # Limit to 10 steps
            step_str = str(detail)[:100] + "..." if len(str(detail)) > 100 else str(detail)
            table.add_row(str(i), step_str)
        
        output = Text()
        output.append("Reason:\n", style="bold yellow")
        output.append(f"{reason[:200]}{'...' if len(reason) > 200 else ''}\n\n", style="white")
        output.append("Summary:\n", style="bold yellow")
        output.append(f"{plan_summary[:200]}{'...' if len(plan_summary) > 200 else ''}", style="white")
        
        self.console.print(Panel(
            output,
            title="[bold yellow]🎯 Planning Agent[/bold yellow]",
            border_style="yellow",
            box=box.DOUBLE
        ))
        
        if plan_detail:
            self.console.print(table)

    def _handle_final_event(self, data: FinalEvent) -> str | None:
        workflow_response = data.workflow_response
        chat_history = data.chat_history

        self.console.print()
        self.console.rule("[bold green]✨ Workflow Complete[/bold green]", style="green")
        self.console.print()
        
        self.console.print(Panel(
            f"[white]{workflow_response}[/white]",
            title="[bold green]🎉 Final Response[/bold green]",
            border_style="green",
            box=box.DOUBLE,
            padding=(1, 2)
        ))
        
        self.console.print(Panel(
            f"[cyan]Total messages in history:[/cyan] [bold]{len(chat_history)}[/bold]\n"
            f"[dim]Session has been updated and saved[/dim]",
            border_style="cyan",
            box=box.ROUNDED
        ))
        
        return workflow_response
        
    
    def _handle_agent_decision(self, data: AgentDecision) -> None:
        """Handle agent decision (branching reasoning)."""
        name = data.name
        decision = data.decision
        reason = data.reason

        output = Text()
        output.append(f"Decision made by: [bold cyan]{name}[/bold cyan]\n\n", style="white")
        output.append(f"[bold]Decision:[/bold] {decision}\n", style="green")
        output.append(f"[bold]Reason:[/bold] {reason}", style="white")

        self.console.print(Panel(
            output,
            title="[bold cyan]🧭 Agent Decision[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        ))


    def _handle_agent_stream_structured_output(self, data: AgentStreamStructuredOutput) ->None:
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

    def _handle_stop(self,data : StopEvent ):
        self.console.print(Panel(
                f"[yellow] Culprit: \n{vars(data)}[/yellow]",
                title="[bold magenta]📊 Structured Output Stream[/bold magenta]",
                border_style="magenta",
                box=box.ROUNDED
            ))
    def _handle_eval(self, data : EvaluationCompleteEvent):
        self.console.print(Panel(
                f"[yellow] Culprit: \n{vars(data)}[/yellow]",
                title="[bold magenta]📊 Structured Output Stream[/bold magenta]",
                border_style="magenta",
                box=box.ROUNDED
            ))
    def handle_event(self, event_data: dict) -> str :
        event_type = event_data['event_type']


        handler_map = {
            UserInputEvent.__name__: self._handle_user_input,
            FinalResponseEvent.__name__: self._handle_final_response,
            PlannerInputEvent.__name__: self._handle_planner_input,
            PlanProposedEvent.__name__: self._handle_plan_proposed,
            AgentProgressEvent.__name__: self._handle_agent_progress,
            AgentDecision.__name__: self._handle_agent_decision,
            PlanningAgentEvent.__name__: self._handle_planning_agent,
            FinalEvent.__name__: self._handle_final_event,
            AgentInput.__name__: self._handle_agent_input,
            AgentOutput.__name__: self._handle_agent_output,
            AgentStream.__name__: self._handle_agent_stream,
            ToolCall.__name__: self._handle_tool_call,
            ToolCallResult.__name__: self._handle_tool_call_result,
            AgentStreamStructuredOutput.__name__: self._handle_agent_stream_structured_output,
            StopEvent.__name__ : self._handle_stop,
            EvaluationCompleteEvent.__name__: self._handle_eval
        }

        handler = handler_map[event_type]
    
        event_payload = {k: v for k, v in event_data.items() if k != 'event_type'}
        event_class_map = {
            UserInputEvent.__name__: UserInputEvent,
            FinalResponseEvent.__name__: FinalResponseEvent,
            PlannerInputEvent.__name__: PlannerInputEvent,
            PlanProposedEvent.__name__: PlanProposedEvent,
            AgentProgressEvent.__name__: AgentProgressEvent,
            AgentDecision.__name__: AgentDecision,
            PlanningAgentEvent.__name__: PlanningAgentEvent,
            FinalEvent.__name__: FinalEvent,
            AgentInput.__name__: AgentInput,
            AgentOutput.__name__: AgentOutput,
            AgentStream.__name__: AgentStream,
            ToolCall.__name__: ToolCall,
            ToolCallResult.__name__: ToolCallResult,
            AgentStreamStructuredOutput.__name__: AgentStreamStructuredOutput,
            StopEvent.__name__ : StopEvent,
            EvaluationCompleteEvent.__name__: EvaluationCompleteEvent
        }
        event_class = event_class_map[event_type]
        instance_event = event_class(**event_payload)
        handler(instance_event)
        return instance_event
