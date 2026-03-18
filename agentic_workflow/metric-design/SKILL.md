---
name: metric-design
description: >-
  Design and implement evaluation metrics for a downstream task, based on
  the data_types and executor from Round 2. May add or modify executors as
  needed. Use after pipeline-generate is confirmed.
---

# Round 3: Metric Design

Round 2 gave you: data_types, loader, and executor(s). Now design and write the evaluation metrics for this downstream task.

## What Drives Metric Design

Metrics are NOT arbitrary code -- they should be grounded in your understanding of:

1. **The dataset itself**: what does each entry contain, what are common quality issues you observed in Round 1
2. **The downstream task**: what is this data used for in post-training (SFT, RLHF, etc.), what capabilities must the model learn from it
3. **What can go wrong**: if a data point is flawed, how would it hurt the trained model? Each metric should catch a specific failure mode

For example, in `api_agent_eval`:
- The task is teaching models to call APIs correctly
- If API names are hallucinated -> model learns to call non-existent APIs -> `format_check` catches this
- If required parameters are missing -> model learns incomplete calls -> `executability` catches this
- If the same query/tool patterns repeat -> model overfits to narrow patterns -> `diversity` catches this
- If multi-step chains have no data dependencies -> model never learns to pass results between steps -> `task_complexity` reveals this

## Two Types of Metrics

### Metrics that need executors (dataset-specific)

These metrics call `executor.check(sample)` because the checking logic differs per dataset. Examples:
- `format_check.py`: calls FormatChecker -- what "correct format" means varies per dataset
- `executability.py`: calls ExecutabilityChecker -- how to validate depends on dataset structure

For these, the metric file (`metrics/format_check.py`) contains the shared orchestration logic (iterate samples, collect stats, save results), and the executor (`executor/<dataset>/FormatChecker.py`) contains the dataset-specific check.

### Metrics that work directly on data_types (shared)

These metrics only use the fixed fields of data_types and apply uniformly to all datasets. Examples:
- `diversity.py`: computes embedding diversity, Self-BLEU, etc. on query/tools/api_calls
- `task_complexity.py`: analyzes tool selection difficulty, parameter filling, planning depth

These do NOT need executors. They read data_types fields directly.

## Step 1: Read All Existing Metrics

Read every metric file in the downstream task's `metrics/` directory. For each one, understand:
- What quality dimension does it measure?
- Does it use an executor or work directly on data_types?
- What are its input parameters and output format?
- How is it wired into `scripts/run_full_test.py`?

## Step 2: Decide What Metrics to Write

Based on your understanding from Round 1 (dataset characteristics, post-training role, similar datasets) and Round 2 (data_types fields available, executor capabilities), decide:

### For Case B (existing task, new dataset):

The existing metrics likely already cover this task. Ask:
- Do all existing metrics work with the new dataset? (They should, if data_types is reused.)
- Does the new dataset have characteristics that existing metrics miss? For example, a new API agent dataset might have response quality issues that current metrics don't check.
- Are there dataset-specific fields in metadata that enable new checks? For example, if the dataset has a `thought` field in metadata, you could add a thought quality metric.

Usually Case B needs zero or few new metrics -- the existing ones should work. But if you identify a gap, write it.

### For Case C (new downstream task):

You need to design metrics from scratch. Think about:
- What are the essential quality dimensions for this task?
- Which dimensions can be checked with shared logic (no executor needed)?
- Which dimensions require dataset-specific executors?

At minimum, most downstream tasks need:
- **format_check**: structural validity of each data point
- **diversity**: how varied the dataset is (often adaptable from existing implementations)

Beyond that, design task-specific metrics based on what failure modes matter for this task's post-training objective.

## LLM Judge and Guard Model Metrics

Not everything can be checked with deterministic rules. Some metrics need LLM or specialized models. Write these alongside your other metrics -- do not defer them.

### LLM-as-Judge for subjective quality

Some quality dimensions are inherently subjective. Use LLM-as-Judge for these. Two patterns exist in the codebase:

1. **Task-universal LLM judge** -- the same judge logic applies to all datasets in a task. Example: `reasoning_validity.py` (at `/modalities/Symbolic_and_Logical_Data/math_eval/metrics/reasoning_validity.py`) uses LLM to validate whether a math reasoning process is logically sound. This works the same for LILA and OpenMath.

2. **Dataset-specific LLM judge via executor constants** -- different datasets need different judge prompts because what "correct" means differs. Examples:
   - ToolBench's `constants.py` (at `/modalities/Agent_Data/api_agent_eval/executor/toolbench/constants.py`) defines `DERIVABILITY_PROMPT` and `RELEVANCE_PROMPT` specific to ToolBench's multi-turn format with final answers
   - xLAM's `constants.py` (at `/modalities/Agent_Data/api_agent_eval/executor/xlam/constants.py`) defines `TOOL_SELECTION_PROMPT` for single-turn tool selection validation

   These prompts are stored in the executor's `constants.py` and called by the executor during metric computation.

When writing LLM judge metrics:
- Design the prompt carefully with clear evaluation criteria and structured JSON output
- Store dataset-specific prompts in `executor/<dataset>/constants.py`
- Write the metric logic with a placeholder for the LLM call, then ask the user for their API configuration (OpenAI key, local model endpoint, etc.) to wire up the actual calls

### Safety and Trustworthiness

Quality is not the only dimension -- safety and trustworthiness also need evaluation. Two approaches exist:

1. **Guard model evaluator** -- use a specialized safety model like AgentDoG (see `/modalities/Agent_Data/api_agent_eval/evaluator/agentdog.py`). The evaluator is loaded separately and passed to the metric function. The metric `trustworthy.py` (at `/modalities/Agent_Data/api_agent_eval/metrics/trustworthy.py`) wraps the guard model evaluation with standard metric output format. Supports binary (safe/unsafe) and fine-grained classification (risk source, failure mode, real world harm).

2. **Rule-based safety checks** -- some safety dimensions can be checked without models (e.g., detecting hardcoded credentials, checking for SQL injection patterns in agent actions, flagging tool calls to sensitive endpoints).

Consider whether this downstream task has safety-relevant actions. If the agent can execute code, call APIs, navigate web pages, or access user data, safety metrics are important.

## Step 3: Write the Metrics

For each new metric, create a file in `metrics/`. Follow the existing pattern:

1. **A `compute_<metric>()` function** that:
   - Takes a `data_iterator` and (optionally) an executor
   - Iterates over samples, applies checks, collects statistics
   - Returns a results dict
   - Saves results to JSON if `output_file` is provided
   - Prints a human-readable summary

2. **A `if __name__ == '__main__':` block** with argparse for standalone testing

3. **Consistent output format**: timestamp, elapsed_seconds, total samples, pass/fail rates, detailed error samples

Read any existing metric file (e.g., `/modalities/Agent_Data/api_agent_eval/metrics/format_check.py`) as the template for this pattern.

## Step 4: Add or Modify Executors (if needed)

If a new metric requires dataset-specific processing that the existing executors don't provide:

1. **Check if an existing executor can be extended** -- maybe adding a method is enough
2. **If not, create a new executor** in `executor/<dataset>/` following the pattern from Round 2
3. **Register it** in `executor/__init__.py`

It is completely fine to go back and modify executor code written in Round 2. The executor and metric design are iterative -- you may realize during metric writing that the executor needs an additional check method.

## Step 5: Wire into run_full_test.py

Add the new metric to `scripts/run_full_test.py`:

1. Add it to the metric choices in the argparse
2. Add the execution block (load data, create checker if needed, call compute function)
3. Make sure `--metric all` includes it

## Step 6: Validate

1. `python -m py_compile` on all new/modified files
2. Run the new metric on a small sample: `python metrics/new_metric.py --dataset <name> --max-samples 10`
3. Check that the output JSON is well-formed and the summary makes sense
4. If any metric uses an executor, test the executor independently too

## Step 7: Report to User

Present:
1. List of metrics created/modified
2. For each metric: what it checks, why it matters for this task, whether it needs an executor
3. Any executor changes made
4. Sample output from a small test run
5. Ready to proceed to Round 4 (audit-run)?

## Step 8: Reflect

Before reporting to the user, stop and think critically:

- **Coverage**: have you thought about this dataset's value from the perspective of what the trained model needs to learn? If a data point is flawed in a way that would hurt the model, is there a metric that catches it? Think about the full chain: data quality -> training signal -> model behavior.
- **Missing angles**: are there quality dimensions you haven't covered? For example, if the task involves multi-step reasoning, did you check trajectory coherence? If the data has natural language instructions, did you check instruction clarity? If the data is synthetic, did you check for template repetition?
- **Executor gaps**: now that you've written the metrics, do any of them need executor support that doesn't exist yet? If so, go back to `executor/` and add it.

If you find gaps, write the additional metrics or executors before reporting.

## What NOT to Do

- Do not write metrics that duplicate what Layer 1 (modality-level) metrics already cover
- Do not create executors for metrics that work directly on data_types fields
- Do not skip reading ALL existing metrics first
- Do not proceed to Round 4 without user confirmation
