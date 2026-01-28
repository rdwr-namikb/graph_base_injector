"""
CLI Interface for Graph-Based Injectr

This module provides the command-line interface for interacting with
Graph-Based Injectr. It uses Typer for argument parsing and Rich for output.

Commands:
    - run: Start an injection testing session
    - test: Quick test of target connectivity
    - list-techniques: List available jailbreak techniques
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.markdown import Markdown

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..config import Settings, get_settings
from ..config.constants import DEFAULT_TARGET_URL, AGENT_MAX_ITERATIONS

# Create Typer app
app = typer.Typer(
    name="graph-based-injectr",
    help="AI-Powered Prompt Injection Testing Framework using LangGraph",
    add_completion=False,
)

# Rich console for output
console = Console()


def print_banner():
    """Print the application banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                    Graph-Based Injectr                         ║
║        AI-Powered Prompt Injection Testing (LangGraph)         ║
╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


@app.command()
def run(
    target: str = typer.Option(
        DEFAULT_TARGET_URL,
        "--target", "-t",
        help="Target LLM server URL",
    ),
    objective: str = typer.Option(
        "",
        "--objective", "-o",
        help="Attack objective (e.g., 'Read /etc/passwd')",
    ),
    max_iterations: int = typer.Option(
        AGENT_MAX_ITERATIONS,
        "--max-iterations", "-m",
        help="Maximum agent iterations",
    ),
    model: str = typer.Option(
        None,
        "--model",
        help="Orchestrator LLM model (default: from settings)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug", "-d",
        help="Enable debug output",
    ),
):
    """
    Start an injection testing session.
    
    This launches the injection agent to test the target LLM for
    prompt injection vulnerabilities.
    
    Example:
        graph-based-injectr run --target http://localhost:8000 --objective "Read /etc/passwd"
    """
    print_banner()
    
    settings = get_settings()
    
    # Override settings with CLI args
    if model:
        settings.model = model
    if debug:
        settings.debug = True
    
    console.print(f"[bold]Target:[/bold] {target}")
    console.print(f"[bold]Objective:[/bold] {objective or 'Explore target capabilities'}")
    console.print(f"[bold]Model:[/bold] {settings.model}")
    console.print()
    
    # Run the agent
    asyncio.run(_run_agent(
        target=target,
        objective=objective,
        max_iterations=max_iterations,
        settings=settings,
    ))


async def _run_agent(
    target: str,
    objective: str,
    max_iterations: int,
    settings: Settings,
):
    """Run the injection agent asynchronously using LangGraph."""
    from ..agents import create_injection_agent, create_injection_initial_state
    from ..llm import create_chat_model
    from ..target import TargetClient
    from ..tools import get_all_tools, set_target_client
    from ..knowledge import RAGEngine
    
    # Initialize components
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing...", total=None)
        
        # Create target client
        progress.update(task, description="Connecting to target...")
        target_client = TargetClient(base_url=target)
        
        # Check target health
        is_healthy = await target_client.health_check()
        if not is_healthy:
            console.print("[red]Warning: Target may not be reachable[/red]")
        
        # Set target client for tools
        set_target_client(target_client)
        
        # Create LangChain chat model
        progress.update(task, description="Initializing LLM...")
        chat_model = create_chat_model(model=settings.model)
        
        # Create RAG engine
        progress.update(task, description="Loading knowledge base...")
        rag = RAGEngine(use_local_embeddings=False)
        try:
            rag.index()
            console.print(f"[dim]Indexed {rag.document_count} knowledge documents[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load knowledge base: {e}[/yellow]")
            rag = None
        
        # Get tools
        tools = get_all_tools()
        
        progress.update(task, description="Creating agent graph...")
        
        # Create the LangGraph agent
        graph = create_injection_agent(
            chat_model=chat_model,
            tools=tools,
            target_client=target_client,
            objective=objective,
            target_model="gpt-5.2",  # From the target description
            target_tools=["read_file", "write_file", "list_files"],
            rag_engine=rag,
            max_iterations=max_iterations,
        )
        
        # Create initial state
        initial_state = create_injection_initial_state(
            objective=objective,
            target_url=target,
            target_model="gpt-5.2",
            target_tools=["read_file", "write_file", "list_files"],
            max_iterations=max_iterations,
            initial_message=objective if objective else "Explore the target's capabilities and test for vulnerabilities",
        )
    
    console.print("\n[bold green]Agent started. Beginning injection testing...[/bold green]\n")
    console.print("-" * 60)
    
    try:
        # Stream events from the graph
        async for event in graph.astream(initial_state):
            # LangGraph returns events with node names as keys
            for node_name, node_output in event.items():
                # Skip if node_output is None
                if node_output is None:
                    continue
                    
                if node_name == "agent":
                    # Agent node output contains messages
                    messages = node_output.get("messages", []) if isinstance(node_output, dict) else []
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            # Check for tool calls
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                # Show tool calls
                                for tc in msg.tool_calls:
                                    tool_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                                    console.print(f"[cyan]→ Calling tool:[/cyan] {tool_name}")
                                    
                                    if tool_name == "send_prompt":
                                        args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                                        prompt_content = args.get("prompt", "")
                                        if prompt_content:
                                            preview = prompt_content[:200] + "..." if len(prompt_content) > 200 else prompt_content
                                            console.print(Panel(preview, title="[dim]Payload[/dim]", border_style="dim"))
                            
                            # Show thinking content if present
                            if msg.content:
                                console.print(Panel(
                                    str(msg.content),
                                    title="[yellow]Agent Thinking[/yellow]",
                                    border_style="yellow",
                                ))
                
                elif node_name == "tools":
                    # Tool node output contains tool results
                    messages = node_output.get("messages", []) if isinstance(node_output, dict) else []
                    for msg in messages:
                        if isinstance(msg, ToolMessage):
                            content = str(msg.content)
                            display_content = content[:500] + "..." if len(content) > 500 else content
                            console.print(Panel(
                                display_content,
                                title=f"[blue]Tool Result[/blue]",
                                border_style="blue",
                            ))
                
                # Check for finish
                if isinstance(node_output, dict) and node_output.get("finished", False):
                    console.print("\n[bold green]Agent completed task[/bold green]")
            
            console.print()
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if settings.debug:
            import traceback
            console.print(traceback.format_exc())
    finally:
        await target_client.close()
    
    console.print("-" * 60)
    console.print("[bold]Session complete[/bold]")


@app.command()
def test(
    target: str = typer.Option(
        DEFAULT_TARGET_URL,
        "--target", "-t",
        help="Target LLM server URL",
    ),
):
    """
    Test connectivity to the target server.
    
    Sends a simple message and displays the response.
    """
    print_banner()
    
    console.print(f"Testing connection to: {target}")
    
    asyncio.run(_test_target(target))


async def _test_target(target: str):
    """Test target connectivity."""
    from ..target import TargetClient
    
    client = TargetClient(base_url=target)
    
    try:
        # Health check
        is_healthy = await client.health_check()
        
        if is_healthy:
            console.print("[green]✓ Health check passed[/green]")
        else:
            console.print("[yellow]⚠ Health check failed (server may still work)[/yellow]")
        
        # Send test message
        console.print("\nSending test message...")
        response = await client.send_message("Hello, what can you help me with?")
        
        if response.success:
            console.print("[green]✓ Got response from target[/green]")
            console.print(Panel(
                response.content[:500] if len(response.content) > 500 else response.content,
                title="Response",
            ))
        else:
            console.print(f"[red]✗ Error: {response.error}[/red]")
            
    finally:
        await client.close()


@app.command("list-techniques")
def list_techniques(
    category: str = typer.Option(
        None,
        "--category", "-c",
        help="Filter by category",
    ),
    min_effectiveness: int = typer.Option(
        0,
        "--min-effectiveness", "-e",
        help="Minimum effectiveness (1-10)",
    ),
):
    """
    List available jailbreak techniques.
    
    Shows all pre-built injection techniques from the payload library.
    """
    from ..payloads import (
        get_all_templates,
        get_templates_by_category,
        JailbreakCategory,
    )
    
    print_banner()
    
    # Get templates
    if category:
        try:
            cat_enum = JailbreakCategory(category)
            templates = get_templates_by_category(cat_enum)
        except ValueError:
            console.print(f"[red]Invalid category. Valid: {[c.value for c in JailbreakCategory]}[/red]")
            return
    else:
        templates = get_all_templates()
    
    # Filter by effectiveness
    if min_effectiveness > 0:
        templates = [t for t in templates if t.effectiveness >= min_effectiveness]
    
    if not templates:
        console.print("[yellow]No techniques match the criteria[/yellow]")
        return
    
    # Display as table
    table = Table(title=f"Jailbreak Techniques ({len(templates)})")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Effectiveness", justify="center")
    table.add_column("Description", max_width=40)
    
    for t in sorted(templates, key=lambda x: -x.effectiveness):
        eff_display = "⭐" * (t.effectiveness // 2)
        table.add_row(
            t.name,
            t.category.value,
            eff_display,
            t.description[:40] + "..." if len(t.description) > 40 else t.description,
        )
    
    console.print(table)


@app.command()
def version():
    """Show version information."""
    from .. import __version__
    console.print(f"Graph-Based Injectr v{__version__}")


def main():
    """Main entry point for the CLI."""
    import warnings
    # Suppress Pydantic serialization warnings caused by LiteLLM response objects
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    # Suppress LiteLLM/LangSmith async cleanup warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="coroutine .* was never awaited")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Enable tracemalloc to get the object allocation traceback")
    
    app()


if __name__ == "__main__":
    main()
