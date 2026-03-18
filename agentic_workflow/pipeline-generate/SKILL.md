---
name: pipeline-generate
description: >-
  Generate data_types, loader, and executor code for a new dataset based on
  Round 1 inspection results. Use after the user confirms the modality match
  and field mapping from data-inspect.
---

# Round 2: Pipeline Code Generation

**Critical rules for this round:**
- Read ALL existing code before writing anything. Every loader, every executor, every .py file in the matched task.
- Do NOT write metrics. Metrics are Round 3.
- The Reflect step at the end is MANDATORY. If you find gaps during reflection, you MUST fix the code before reporting. Do not just acknowledge the gap -- fix it.
- Do NOT proceed to Round 3 until the user explicitly confirms.

Round 1 gave you: modality, downstream task (Case A/B/C), field mapping, metrics analysis. Now write the code.

## Prerequisites

Before starting, confirm you have from Round 1:
- Which modality (one of the 6 fixed ones)
- Which case: A (existing task + existing dataset), B (existing task + new dataset), or C (new downstream task)
- The field mapping: which user fields -> fixed fields, which -> metadata
- The metrics compatibility analysis

If Case A: no code generation needed, skip to Round 3 (audit-run). Otherwise continue.

## Step 1: Read Reference Code

Based on the matched modality and downstream task, read the relevant existing implementations. You need to understand the patterns before writing anything.

### For Case B (existing task, new dataset):

Read the downstream task you are adding to:

1. **data_types.py** -- you will reuse this, do NOT modify it
   - e.g. `/modalities/Agent_Data/api_agent_eval/data_types.py`

2. **loaders.py** -- read ALL existing loaders in this file to understand the pattern
   - How `BaseLoader` works (iterate vs parse_all)
   - How each dataset loader maps its raw format to the data_types
   - How metadata is populated with dataset-specific fields
   - e.g. `/modalities/Agent_Data/api_agent_eval/loaders.py` contains ToolBenchLoader, XLAMLoader, ArceeAgentDataLoader, GeneralLoader

3. **executor/__init__.py** -- understand the registration pattern
   - How checkers are imported and registered via `register_format_checker()`, `register_executability_checker()`
   - e.g. `/modalities/Agent_Data/api_agent_eval/executor/__init__.py`

4. **ALL existing dataset executors in this downstream task** -- read every checker, every .py file
   - e.g. for api_agent_eval, read all of:
     - `/modalities/Agent_Data/api_agent_eval/executor/toolbench/` -- all .py files
     - `/modalities/Agent_Data/api_agent_eval/executor/xlam/` -- all .py files
     - `/modalities/Agent_Data/api_agent_eval/executor/arcee_agent/` -- all .py files
     - `/modalities/Agent_Data/api_agent_eval/executor/general/` -- all .py files
   - **Why read all of them**: each dataset's executor reflects that dataset's unique characteristics. For example:
     - ToolBench has multi-turn conversations with Thought/Action/Action Input, so its FormatChecker validates thought structure and Finish action format
     - xLAM is single-turn with tool definitions, so its FormatChecker checks parameter schema annotations and type consistency
     - Arcee mixes 5 sub-formats, so its checker routes by sub-dataset
   - By reading all of them, you understand what kind of dataset-specific checks are possible and at what granularity. Then when you look at the NEW dataset's unique characteristics (its format quirks, special fields, edge cases you saw in Round 1), you can design an executor that addresses those specifics rather than writing a generic checker that misses the point

5. **scripts/run_full_test.py** -- understand the DATASETS config and how datasets are registered
   - Where data paths, loader classes, and checker names are configured

### For Case C (new downstream task):

Read everything above, PLUS:

6. **All existing data_types.py files** -- read them to understand the design pattern (fixed fields + metadata, dataclass style, helper methods). You are not copying any specific one; you are learning the convention so you can write a data_types.py that fits this project's style while accurately capturing the NEW task's essential characteristics.
7. **The modality-level metrics** at `modalities/<modality>/metrics/` -- these are Layer 1 metrics that audit individual data points before they form a dataset. They include checks for both synthetic and natural data (e.g., validity, safety, fidelity). Not all of them apply to every dataset -- read each one and judge which are relevant based on the data's nature. Your data_types should provide the fields that the relevant Layer 1 metrics need to run, but you do not need to accommodate metrics that don't apply.
8. **All existing task-level metrics you can find** -- not to copy, but to understand the depth and granularity of checks this project expects. Then, based on your understanding of the new dataset, its role in post-training, and its unique characteristics, think about what Layer 2 metrics this new task will eventually need. Your data_types design should make those future metrics possible (e.g., if you foresee needing a "code executability" metric, make sure the solution field is accessible, not buried in metadata)

## Step 2: Write the Loader

Add a new loader class to the existing `loaders.py` file. Follow these rules:

1. **Inherit from BaseLoader** in the same file
2. **Map every user field** to either a fixed field or metadata, as planned in Round 1
3. **Handle the file format** (json/jsonl/gz/parquet) correctly
4. **Generate sample_id** if the dataset does not have one
5. **Populate metadata** with dataset-specific fields that might be useful for metrics

Example pattern (from existing loaders):
```python
class NewDatasetLoader(BaseLoader):
    """
    NewDataset loader.
    
    Raw format:
    {
        "user_field_1": "...",
        "user_field_2": [...],
        ...
    }
    """
    def iterate(self) -> Iterator[SampleType]:
        for idx, raw in enumerate(self._read_file()):
            yield SampleType(
                fixed_field_1=raw['user_field_1'],
                fixed_field_2=self._parse_field_2(raw['user_field_2']),
                sample_id=f"newdataset_{idx}",
                metadata={
                    'dataset_specific_field': raw.get('extra_field'),
                },
            )
```

Always include a docstring that shows the raw data format so future readers understand the mapping.

## Step 3: Write the Executor (only if needed)

**Not every dataset needs an executor.** Understand the distinction:

- **Metrics** (in `metrics/`) contain the evaluation logic that applies to ALL datasets in a downstream task. For example, `diversity.py` and `task_complexity.py` in `api_agent_eval/metrics/` work for ToolBench, xLAM, and Arcee equally -- they only depend on the shared `data_types` fields (query, tools, api_calls). These metrics do NOT need executors.

- **Executors** (in `executor/<dataset>/`) are only needed when a metric requires **dataset-specific processing** that differs between datasets. For example:
  - `format_check.py` calls `format_checker.check(sample)` -- but what "correct format" means differs per dataset (ToolBench checks Thought/Action structure, xLAM checks parameter schema). So each dataset has its own FormatChecker executor.
  - `executability.py` calls `executability_checker.check(sample)` -- but how to validate API calls differs (ToolBench matches against a local toolenv, xLAM checks against inline tool definitions, Arcee uses LLM judge). So each dataset has its own ExecutabilityChecker executor.
  - For math, answer extraction differs per dataset (LILA executes Python code, OpenMath parses boxed LaTeX). So each has its own CodeExtractor and ResultComparator executor.

The pattern is: **metrics define WHAT to check, executors define HOW to check for a specific dataset**. The metric function takes an executor as a parameter:

```python
# metric (shared logic)
results = compute_format_check(
    data_iterator=loader.iterate(),
    format_checker=checker,        # <-- dataset-specific executor passed in
)
```

So ask yourself: for each existing metric in this downstream task, does the new dataset need a different implementation of the check? If yes, write an executor. If the default/general checker works, skip it.

Create a new executor directory only if needed: `executor/<dataset_name>/`

Files to create:

1. **`__init__.py`** -- import and export the checker classes
2. **`<Dataset>FormatChecker.py`** -- dataset-specific format checks
3. **`<Dataset>ExecutabilityChecker.py`** (if applicable) -- dataset-specific validity checks
4. **`constants.py`** (if needed) -- prompts, thresholds, field mappings

Every real-world dataset needs its own FormatChecker. The GeneralFormatChecker only exists for datasets that were constructed exactly according to our data_types spec with no extra structure -- this is rare and not the normal case.

Your FormatChecker should check things specific to THIS dataset's format and structure. For example:
- ToolBench's FormatChecker checks thought format, Action/Action Input structure, and Finish action semantics
- xLAM's FormatChecker checks tool definition schema, parameter annotations, and type consistency
- Arcee's FormatChecker routes by sub-format because it contains 5 different dataset styles

Think about what can go wrong in this specific dataset's format, and check for it.

## Step 4: Write data_types.py (Case C only)

Only if creating a new downstream task. Follow the existing pattern:

1. Use `@dataclass` with clear field documentation
2. **Fixed fields**: the universal fields shared by ALL datasets in this task (identified in Round 1 Step 4)
3. **`metadata: Dict[str, Any]`**: always include this as escape hatch for dataset-specific fields
4. Include `__repr__` for readable debug output
5. Include helper methods if they make sense (e.g., `get_tool_names()` in APIAgentSample)

Read `/modalities/Agent_Data/api_agent_eval/data_types.py` as the gold standard example.

## Step 5: Register

### Register the new dataset in the executor __init__.py:

Add import and registration lines following the existing pattern:
```python
from .new_dataset import (
    NewDatasetFormatChecker,
    NewDatasetExecutabilityChecker,
)
register_format_checker('new-dataset', NewDatasetFormatChecker)
register_executability_checker('new-dataset', NewDatasetExecutabilityChecker)
```

### Register in scripts/run_full_test.py DATASETS config:

Add an entry with the dataset name, data path, loader class, and checker names.

## Step 6: Validate

1. Run `python -m py_compile` on every file you created or modified
2. Try importing the new loader: `python -c "from loaders import NewDatasetLoader"`
3. Try loading one sample: verify it produces a valid data_types instance
4. If any step fails, fix the error before reporting

## Step 7: Report to User

Present:
1. List of files created/modified
2. The data_types mapping (show how user fields map to fixed fields + metadata)
3. What the executor checks
4. Any concerns or limitations
5. Ready to proceed to Round 3 (audit-run)?

## Step 8: Reflect

Before reporting to the user, go through this checklist. For each item, if the answer is "no", go back and fix it NOW. Do not just report the gap -- fix it.

- [ ] **data_types completeness**: are the fixed fields truly universal for this downstream task? Did you miss any field that multiple similar datasets share? Is the metadata dict capturing all dataset-specific fields that might be useful for future metrics?
- [ ] **Loader edge cases**: does it handle empty fields, missing keys, malformed entries? Did you test entries from different positions in the file (not just the first few)?
- [ ] **Executor coverage**: does the FormatChecker cover ALL structural issues you observed in the data during Round 1? Are there format quirks you noticed but didn't write a check for?
- [ ] **ExecutabilityChecker thoroughness**: does it validate everything that "correct" means for this dataset? Did you miss any validation dimension?

If ANY item above is not satisfied, fix the code before proceeding to report.

## What NOT to Do

- Do not modify existing data_types.py (Case B). If the field mapping does not work, rethink the mapping, not the data_types.
- Do not write metrics in this round. Metrics are designed in Round 3 (metric-design).
- Do not skip reading existing code. Every new file should be consistent with the codebase style.
- Do not create a new modality directory. Modalities are fixed.
- Do not proceed to Round 3 without user confirmation.
