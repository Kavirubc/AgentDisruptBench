# NeurIPS 2026: Evaluations & Datasets (ED) Track Guidelines

> **Notice to Co-authors**: The NeurIPS 2026 Datasets & Benchmarks Track has been officially renamed to the **Evaluations & Datasets (ED) Track**. 
> As a result, the emphasis of our paper needs to shift. We are no longer just publishing a "benchmark"; we are introducing an **Evaluation Methodology** as a scientific object of study.

When drafting the paper (Introduction, Methodology, and Conclusion), please ensure the following points from the NeurIPS CFP update are heavily emphasized:

### 1. Evaluation as a Scientific Object
The core of our paper is *not* just "we made 100 tasks and ran GPT-4o". 
**The core contribution is the methodology itself**: 
We identified that current benchmarks fail because they assume perfect environments. We have built an evaluation methodology — consisting of the Disruption Engine, the $R(k, \epsilon, \lambda)$ surface, and strategy classifications — to properly *study* how LLM agents behave when things break.

### 2. Don't Just "Beat a Baseline"
NeurIPS explicitly stated: *“A submission need not beat a baseline; its primary contribution should be to deepen and refine our understanding of evaluation practices.”*
Our paper should focus heavily on the figures that compare **Recovery Strategies** and **Degradation**. The raw "Success Rate" is less important than *why* and *how* the models fail and recover. 

### 3. Clear Terminology
- Avoid saying: "We present a new dataset of 100 tasks."
- Instead say: "We present a comprehensive evaluation methodology leveraging 100 base tasks and variants..."
- Avoid saying: "AgentDisruptBench is a benchmark..."
- Instead say: "AgentDisruptBench is an evaluation framework/methodology..."

### 4. Justifying the Mocked Sandbox
It is critical to defend why a simulated, mocked wrapper (instead of live endpoints or WebArena-style full-stack deployments) is scientifically valid and necessary:
- **Perfect Determinism**: Live endpoints experience network drift and uncontrolled variability, making comparative robustness research impossible to reproduce. Simulated tools allow us to isolate the agent's behavior.
- **Reproducible Chaos Engineering**: By using a `ToolProxy`, we can inject mathematically exact percentages of failures (e.g., exactly 12% timeout rate on `book_flight`). This rigorous, controlled injection is what bridges academics and actual site reliability engineering (SRE).

### 5. Code & Metadata Updates
The `AgentDisruptBench` repository metadata (README, Datasheets, HuggingFace Cards, and Croissant metadata) have all been automatically updated to reflect this new naming convention and focus. When referring to the repository in the paper, refer to it as an "evaluation framework" rather than just a "dataset".
