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
    task_id: str = typer.Option(None, help="Specific task ID to run (e.g., 'retail_001')"),
    tasks: int = typer.Option(10, help="Number of tasks to execute"),
    domain: str = typer.Option(None, help="Filter tasks by domain (e.g., 'standard', 'adversarial')"),
    profile: str = typer.Option("clean", help="Disruption profile to evaluate against"),
    server_url: str = typer.Option("http://localhost:8080", help="URL of the running Sandbox Server")
):
    """Run an evaluation over the benchmark tasks using a specified reference client."""
    typer.echo(f"Evaluating framework '{framework}' over {tasks} tasks using profile '{profile}' against {server_url}...")
    
    import os
    import sys
    # Ensure local 'evaluation' python directory is importable
    local_eval_path = os.path.abspath(os.path.join(os.getcwd()))
    if local_eval_path not in sys.path:
        sys.path.insert(0, local_eval_path)
    
    from evaluation.orchestrator import SandboxOrchestrator
    orchestrator = SandboxOrchestrator(server_url=server_url)
    
    if not orchestrator.check_server():
        raise typer.Exit(code=1)
        
    client_runner = None
    if framework.lower() == "langchain":
        from evaluation.clients.langchain_rest_client import build_langchain_client
        client_runner = build_langchain_client(server_url)
    else:
        typer.echo(f"Framework '{framework}' not yet implemented.", err=True)
        raise typer.Exit(code=1)
        
    report_file = orchestrator.run_evaluation(
        agent_client_func=client_runner,
        profile=profile,
        num_tasks=tasks,
        domain=domain,
        task_id=task_id
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


@app.command()
def dashboard(
    run_id: str = typer.Option(None, help="Specific run ID to load, defaults to latest")
):
    """Launch the interactive AgentDisruptBench local web dashboard."""
    import subprocess
    import os
    dashboard_script = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "dashboard.py")
    dashboard_script = os.path.abspath(dashboard_script)
    
    cmd = ["streamlit", "run", dashboard_script]
    if run_id:
        # Note: We can expand dashboard.py to accept arguments later, for now we just launch it.
        pass
        
    typer.echo(f"Starting Sandstorm Dashboard...")
    subprocess.run(cmd)

@app.command()
def proxy(
    port: int = typer.Option(8082, help="Port to run the OpenAI-compatible proxy on"),
    sandbox_url: str = typer.Option("http://localhost:8080", help="URL of the running Sandbox Server"),
):
    """Launch the AgentDisruptBench Proxy Server."""
    import os
    import uvicorn
    
    # Pass sandbox URL to the proxy app via env var
    os.environ["ADB_SANDBOX_URL"] = sandbox_url
    
    typer.echo(f"Starting Agent Benchmark OpenAI Proxy Server on port {port}...")
    typer.echo(f"  Sandbox target: {sandbox_url}")
    
    uvicorn.run(
        "agentdisruptbench.server.proxy:app", 
        host="127.0.0.1", 
        port=port, 
        log_level="info",
    )


@app.command()
def start(
    server_port: int = typer.Option(8081, help="Port for the Sandbox Server"),
    proxy_port: int = typer.Option(8082, help="Port for the OpenAI Proxy"),
    dashboard_port: int = typer.Option(8501, help="Port for the Dashboard"),
):
    """Start the entire AgentDisruptBench stack (Server, Proxy, Dashboard) in the background."""
    import subprocess
    import sys
    import os
    import time

    # Use the current python executable to ensure we stay in the same venv
    python_exe = sys.executable
    
    typer.echo("🚀 Starting AgentDisruptBench Stack...")

    # 1. Start Sandbox Server
    typer.echo(f"  - Starting Sandbox Server on port {server_port}...")
    subprocess.Popen(
        [python_exe, "-m", "agentdisruptbench.cli", "serve", "--port", str(server_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    
    # 2. Start Proxy
    typer.echo(f"  - Starting OpenAI Proxy on port {proxy_port} (Targeting {server_port})...")
    proxy_env = os.environ.copy()
    proxy_env["ADB_SANDBOX_URL"] = f"http://localhost:{server_port}"
    subprocess.Popen(
        [python_exe, "-m", "agentdisruptbench.cli", "proxy", "--port", str(proxy_port), "--sandbox-url", f"http://localhost:{server_port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=proxy_env,
        start_new_session=True
    )
    
    # 3. Start Dashboard
    typer.echo(f"  - Starting Dashboard on port {dashboard_port}...")
    dashboard_script = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "dashboard.py")
    dashboard_script = os.path.abspath(dashboard_script)
    subprocess.Popen(
        [python_exe, "-m", "streamlit", "run", dashboard_script, "--server.port", str(dashboard_port), "--server.headless", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )

    time.sleep(2)
    typer.echo("\n✅ All systems launched!")
    typer.echo(f"  - Server:    http://localhost:{server_port}")
    typer.echo(f"  - Proxy:     http://localhost:{proxy_port}/v1")
    typer.echo(f"  - Dashboard: http://localhost:{dashboard_port}")
    typer.echo("\nRun 'adb stop' to shut everything down.")


@app.command()
def stop():
    """Stop all running AgentDisruptBench processes."""
    import subprocess
    typer.echo("🛑 Stopping AgentDisruptBench services...")
    # Kill adb processes and streamlit
    subprocess.run(["pkill", "-f", "agentdisruptbench.cli"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "streamlit run"], stderr=subprocess.DEVNULL)
    typer.echo("Done.")


if __name__ == "__main__":
    app()
