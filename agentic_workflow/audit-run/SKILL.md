---
name: audit-run
description: >-
  Run the full evaluation pipeline on the dataset using the code from
  Rounds 2-3. Collect results, analyze findings, and present the audit
  report. Use after metric-design is confirmed.
---

# Round 4: Run Audit

Rounds 2-3 gave you: data_types, loader, executor(s), and metrics. Now run everything and produce the audit report.

## Step 1: Verify Everything is Registered

Before running, confirm:

1. The new dataset is in `scripts/run_full_test.py` DATASETS config with correct:
   - data path
   - loader class
   - checker/executor names
2. All executors are registered in `executor/__init__.py`
3. All metrics are wired into `run_full_test.py` (including in the `--metric all` path)

## Step 2: Run Layer 1 Metrics (Modality-Level)

Run the modality-level metrics that apply to this dataset. These are in `modalities/<modality>/metrics/`.

Not all Layer 1 metrics apply to every dataset -- use the analysis from Round 1 Step 3c to decide which ones to run.

## Step 3: Run Layer 2 Metrics (Task-Level)

Run all task-level metrics using the unified entry point:

```bash
# Run all metrics at once
python evaluate.py <modality> <dataset> all

# Or submit as a job
PARTITION=t-cpu-new NUM_GPUS=0 CPUS=16 ./submit_eval.sh <modality> <dataset> all
```

If some metrics need GPU (e.g., diversity with large embedding models):
```bash
PARTITION=TDS NUM_GPUS=1 ./submit_eval.sh <modality> <dataset> diversity
```

Monitor the job and check for errors in the log files at `results/<dataset>/`.

## Step 4: Collect and Analyze Results

Read all result files from `results/<dataset>/`:
- `format_check_results.json`
- `executability_results.json`
- `diversity_results.json`
- (and any other metrics)

For each metric, note:
- Pass/fail rates
- Common error types
- Distribution patterns
- Anything surprising or concerning

## Step 5: Present Audit Report

Summarize findings for the user:

1. **Dataset overview**: name, size, modality, downstream task
2. **Layer 1 results**: which modality-level checks were run, what they found
3. **Layer 2 results**: for each metric, the key numbers and what they mean
4. **Quality issues found**: specific problems, their severity, how many samples affected
5. **Recommendations**: what to filter, what to fix, whether the dataset is usable for its intended purpose
6. **Comparison** (if possible): how does this dataset compare to similar datasets already evaluated in the framework

## Step 6: Reflect -- Validate Results Against Real Data

Do NOT just trust the numbers. Re-read the dataset from scratch with the metric results in hand, and cross-validate:

1. **Re-read the dataset**: go back and read the actual data entries again -- not from memory of Round 1, but fresh. This time you have the metric results, so read with specific questions: "the format_check says 15% error rate -- do I see format issues when I read random entries?" Read entries from beginning, middle, and end of the file.

2. **Read ALL error samples**: for each metric that reports errors, read every error sample in the result JSON. For each one, go back to the raw data entry and verify: is this a real problem, or is the metric/executor wrong?

3. **Read passing samples with suspicion**: read at least 20-30 samples that passed all checks. Look specifically for problems that SHOULD have been caught but weren't. Are there obvious quality issues the metrics missed?

4. **Cross-check distributions against data reality**: if a metric reports an unusual distribution (e.g., 99% pass rate on format but 40% fail on executability), does that match what you see when browsing the actual data? If not, something is off.

5. **Check for blind spots**: compare what you observe in the data now with what the metrics report. Are there quality problems visible in the data that do NOT show up in any metric result? If so, the metrics have a gap.

If the results do not match reality:
- If a metric flags things that are not real problems -> the executor or metric logic has a bug -> go back and fix
- If real problems are not flagged -> the metrics are incomplete -> go back to Round 3 and add coverage
- If the numbers seem right but the interpretation is unclear -> add context from your data reading to the report

Only present the report after you are confident the results reflect the actual state of the data.

## What NOT to Do

- Do not skip Layer 1 metrics
- Do not ignore errors in the log files -- if a metric fails, debug it before reporting
- Do not present raw numbers without interpretation -- the user needs to understand what the numbers mean for their use case
- Do not trust metric results blindly -- always validate against real data samples
