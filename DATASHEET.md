# Datasheet for AgentDisruptBench

_Following the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) framework (Gebru et al., 2021)._

---

## Motivation

### For what purpose was the dataset created?

AgentDisruptBench was created to study **evaluation as a scientific object** in the context of AI agent resilience under runtime tool-call disruptions. Existing benchmarks measure *whether* agents can use tools correctly, but assume tools behave perfectly — an unrealistic assumption in production environments. This project provides an evaluation methodology, consisting of structured tasks paired with a Disruption Engine, to rigorously measure how we assess agent reliability, recovery strategies, and graceful degradation using the $R(k, \epsilon, \lambda)$ surface.

### Who created the dataset and on behalf of which entity?

AgentDisruptBench Contributors, as part of academic research into AI agent robustness.

### Who funded the creation of the dataset?

<!-- Fill in: grant numbers, institutional support, etc. -->
[To be specified]

---

## Composition

### What do the instances that comprise the dataset represent?

Each instance is a **benchmark task** defined in YAML with:
- A natural language task description (the prompt given to the agent)
- Required tools (from a set of 30 deterministic simulated tools, 6-8 per domain)
- Ground truth: expected outcome, required tool calls, evaluation rubric with weighted criteria
- Metadata: domain, difficulty (1-5), task type

### How many instances are there in total?

**100 tasks** organized as follows:

| Domain     | Standard | Adversarial | Impossible | Handover | Total |
|------------|:--------:|:-----------:|:----------:|:--------:|:-----:|
| Retail     | 20       | 2           | 2          | 1        | 25    |
| Travel     | 20       | 2           | 2          | 1        | 25    |
| Finance    | 20       | 2           | 2          | 1        | 25    |
| DevOps     | 20       | 2           | 2          | 1        | 25    |
| **Total**  | **80**   | **8**       | **8**      | **4**    | **100** |

### Does the dataset contain all possible instances or is it a sample?

The dataset is a curated collection of synthetic tasks. It is not a sample from a larger population. Task difficulty levels are distributed as D1=8, D2=16, D3=24, D4=20, D5=12 tasks across standard tasks, following a roughly bell-shaped distribution centered on medium difficulty.

### What data does each instance consist of?

Each task instance contains:
- `task_id`: Unique identifier (e.g., `retail_001`)
- `title`: Short human-readable title
- `description`: Full prompt given to the agent (natural language)
- `domain`: One of {retail, travel, finance, devops}
- `difficulty`: Integer 1-5
- `task_type`: One of {standard, adversarial, impossible}
- `required_tools`: List of tool names needed
- `expected_tool_call_depth`: Expected number of tool calls under clean conditions
- `ground_truth`: Structured evaluation criteria including:
  - `expected_outcome`: Description of success
  - `required_tool_calls`: Tools that must be called
  - `forbidden_tool_calls`: Tools that must NOT be called
  - `correct_final_answer`: Exact expected answer (if applicable)
  - `evaluation_rubric`: Criterion → weight mapping (sums ≈ 1.0)
  - `disruption_sensitive_tools`: Tools where failure is most impactful
  - `recovery_actions`: Expected recovery behaviours
  - `trap_description`: (adversarial only) the trap the agent should avoid
  - `impossibility_reason`: (impossible only) why no valid solution exists

### Is any information missing from individual instances?

No. All fields are populated for every task instance.

### Are relationships between individual instances made explicit?

Tasks within the same domain share tools and simulated data (e.g., retail tasks all use the same product catalog and customer database via deterministic simulated tools). Cross-domain relationships do not exist.

### Are there recommended data splits?

No train/test split is provided because this is a benchmark, not a training dataset. All 100 tasks are intended for evaluation. However, researchers may partition by domain, difficulty, or task type for focused analysis.

### Does the dataset contain data that might be considered confidential?

No. All data is synthetic. No real personal data, API keys, or proprietary information is included.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

No. Tasks involve routine operations (e-commerce, travel booking, financial transactions, DevOps operations) with no offensive content.

---

## Collection Process

### How was the data associated with each instance acquired?

All tasks were authored manually by the benchmark contributors. They are synthetic, designed to cover a range of realistic tool-use scenarios across difficulty levels.

### What mechanisms or procedures were used to collect the data?

Task design followed a structured process:
1. Define 4 domains based on common LLM agent use cases
2. Design 6-8 simulated tools per domain (30 total)
3. Author 20 standard tasks per domain with increasing difficulty
4. Author adversarial tasks with designed trap patterns
5. Author impossible tasks with clear impossibility reasons
6. Author handover tasks requiring human escalation
7. Validate all ground truth against simulated tool outputs

### Who was involved in the data collection process?

The benchmark authors.

### Over what timeframe was the data collected?

March-April 2026.

### Were any ethical review processes conducted?

<!-- Fill in if applicable -->
[To be specified]

---

## Uses

### What are the intended uses of the dataset?

1. **Primary**: Evaluate AI agent resilience under runtime tool-call disruptions using the provided disruption profiles (clean → hostile_environment)
2. **Secondary**: Compare agent frameworks (LangChain, OpenAI, AutoGen, CrewAI) on reliability metrics
3. **Tertiary**: Study recovery strategies and failure patterns in LLM agents

### What are some tasks/uses that the dataset should not be used for?

- **Not for training**: The benchmark should be used for evaluation only. Training on the tasks would compromise benchmark validity.
- **Not for real tool testing**: Simulated tools are deterministic simulations, not real API integrations.
- **Not for safety/security evaluation**: The disruptions model reliability failures, not adversarial attacks.

### Is there anything about the composition of the dataset or the way it was collected that might impact future uses?

- Tasks are in English only
- Simulated tools simulate US-centric services (USD currency, US dates, etc.)
- Ground truth rubrics use string-matching heuristics which may not capture all valid agent responses

---

## Distribution

### How will the dataset be distributed?

- **Primary**: GitHub repository at [https://github.com/Kavirubc/AgentDisruptBench](https://github.com/Kavirubc/AgentDisruptBench)
- **Persistent**: HuggingFace Datasets at [To be created]
- **Metadata**: Croissant JSON-LD file included in the repository

### When will the dataset be distributed?

The dataset is publicly available as of the initial release.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license?

MIT License.

### Have any third parties imposed IP-based or other restrictions on the data?

No.

---

## Maintenance

### Who is supporting/hosting/maintaining the dataset?

AgentDisruptBench Contributors via the GitHub repository.

### How can the owner/curator/manager of the dataset be contacted?

Via GitHub Issues at the repository.

### Will the dataset be updated?

Yes. Planned updates include:
- Additional task variants for ε-robustness testing
- New domains

### If the dataset relates to people, are there applicable limits on the retention of the data?

N/A — the dataset is entirely synthetic with no real personal data.

### Will older versions of the dataset continue to be available?

Yes, via Git tags and versioned releases.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for doing so?

Yes. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing new tasks, domains, tools, and disruption types.
