import csv
import json
import re
import sys
import os
from pathlib import Path

# Add project root to path so we can import the framework
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, project_root)

from agentdisruptbench import TaskRegistry

def main():
    # Load tasks from YAML
    print("Loading tasks from YAML definitions...")
    registry = TaskRegistry.from_builtin()
    task_map = {t.task_id: {"desc": t.description, "difficulty": t.difficulty} for t in registry.all_tasks()}

    runs_to_process = [
        "20260403_060839_langchain_gemini25flash_all",
        "20260403_062610_langchain_gemini25flash_all",
        "20260403_064525_langchain_gpt5mini_all",
        "20260403_073017_langchain_gpt5mini_all"
    ]
    
    runs_dir = Path("runs")
    all_tasks = []
    
    for rid in runs_to_process:
        d = runs_dir / rid
        jsonl = d / "run_log.jsonl"
        if not jsonl.exists():
            continue
            
        model = "?"
        runner = "?"
        profile = "?"
        
        events = []
        with open(jsonl) as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
                    
        for e in events:
            if e["event_type"] == "run_started":
                model = e["payload"].get("model", "?")
                runner = e["payload"].get("runner", "?")
                profile = e["payload"].get("profile", "?")
                
        # Get tasks
        for e in events:
            if e["event_type"] == "task_completed":
                p = e["payload"]
                task_id = p.get("task_id", "")
                
                task = {
                    "Run_ID": rid,
                    "Model": model,
                    "Runner": runner,
                    "Profile": profile,
                    "Task_ID": task_id,
                    "Difficulty": task_map.get(task_id, {}).get("difficulty", 0),
                    "Description": task_map.get(task_id, {}).get("desc", ""),
                    "Success": 1 if p.get("success", False) else 0,
                    "Partial_Score": p.get("partial_score", 0.0),
                    "Recovery_Rate": p.get("recovery_rate", 0.0),
                    "Total_Tool_Calls": p.get("total_tool_calls", 0),
                    "Disruptions_Encountered": p.get("disruptions_encountered", 0),
                    "Duration_Seconds": p.get("duration_seconds", 0.0),
                }
                all_tasks.append(task)
                
    if not all_tasks:
        print("No tasks found to export.")
        return
        
    out_path = Path("paper/tables/metrics_all_tasks.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", newline="") as f:
        # Use fieldnames directly from the dict to ensure correct column order
        fieldnames = list(all_tasks[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_tasks)
        
    print(f"✅ Exported {len(all_tasks)} rows to {out_path}")

if __name__ == "__main__":
    main()
