"""
AgentDisruptBench — Unified CLI
===============================

File:        cli.py
Purpose:     The single executable entrypoint for the AgentDisrupt-Bench framework.
             Provides commands for serving the sandbox environment and orchestrating evaluations.
"""

import typer
import logging
import sys

logger = logging.getLogger("agentdisruptbench.cli")

app = typer.Typer(
    help="AgentDisrupt-Bench: The standard sandbox for testing agent resilience in hostile environments.",
    no_args_is_help=True
)


@app.command()
def serve(
    mode: str = typer.Option("rest", help="'rest' for FastAPI OpenAPI Server, or 'mcp' for Model Context Protocol Server"),
    port: int = typer.Option(8080, help="Port to run the server on (for REST mode)"),
    profile: str = typer.Option("clean", help="Disruption profile to load"),
    seed: int = typer.Option(42, help="Random seed for disruptions")
):
    """Start the Standard Sandbox server environment."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    
    if mode.lower() == "mcp":
        import agentdisruptbench.server.mcp_server as mcp_server
        typer.echo(f"Starting MCP Server with profile='{profile}' and seed={seed} (Transport: stdio)...")
        # Initialize and run
        server = mcp_server.MCPBenchmarkServer()
        server.setup_run(profile=profile, seed=seed)
        server.run_stdio()
        
    elif mode.lower() == "rest":
        import uvicorn
        typer.echo(f"Starting FastAPI Server on port {port} with profile='{profile}' and seed={seed}...")
        
        # We start the server. The initial setup happens dynamically when hit or via admin.
        uvicorn.run(
            "agentdisruptbench.server.app:app", 
            host="127.0.0.1", 
            port=port, 
            log_level="info",
            # FastAPI expects setup via the admin endpoints, so we do it via a quick client request
            # or just load default config in app.py
        )
    else:
        typer.echo(f"Unknown mode: {mode}. Please use 'rest' or 'mcp'.", err=True)
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    framework: str = typer.Option("langchain", help="Which reference client framework to evaluate (e.g., 'langchain')"),
    tasks: int = typer.Option(10, help="Number of tasks to execute"),
    profile: str = typer.Option("clean", help="Disruption profile to evaluate against"),
    server_url: str = typer.Option("http://localhost:8080", help="URL of the running Sandbox Server")
):
    """Run an evaluation over the benchmark tasks using a specified reference client."""
    typer.echo(f"Evaluating framework '{framework}' over {tasks} tasks using profile '{profile}' against {server_url}...")
    
    from agentdisruptbench.evaluation.orchestrator import SandboxOrchestrator
    orchestrator = SandboxOrchestrator(server_url=server_url)
    
    if not orchestrator.check_server():
        raise typer.Exit(code=1)
        
    client_runner = None
    if framework.lower() == "langchain":
        from agentdisruptbench.evaluation.clients.langchain_rest_client import build_langchain_client
        client_runner = build_langchain_client(server_url)
    else:
        typer.echo(f"Framework '{framework}' not yet implemented.", err=True)
        raise typer.Exit(code=1)
        
    report_file = orchestrator.run_evaluation(
        agent_client_func=client_runner,
        profile=profile,
        num_tasks=tasks
    )
    typer.echo(f"Evaluation complete! Report written to: {report_file}")
    

@app.command()
def report(
    run_id: str = typer.Option(..., help="The Run ID or Path to generate a report for")
):
    """Generate benchmarking metrics and a markdown report for a completed run."""
    typer.echo(f"Generating report for run: {run_id}...")
    from agentdisruptbench.harness.reporter import generate_summary_report
    generate_summary_report(run_id)


if __name__ == "__main__":
    app()
