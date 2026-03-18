---
name: data-inspect
description: >-
  Inspect a user-provided dataset, detect its modality and downstream task,
  find similar open-source datasets, and propose field mappings to the
  evaluation framework. Use when the user gives a dataset path and wants to
  start the agentic data audit workflow.
---

# Round 1: Data Inspection & Modality Detection

**Critical rules for this round:**
- Read the data THOROUGHLY. Do not skim. Read from beginning, middle, and end.
- Do NOT write any code or create any files.
- Do NOT proceed to Round 2 until the user explicitly confirms.

User gives you a dataset path. Your job: deeply understand the data, determine its modality and downstream task, then tell the user how it maps to the evaluation framework.

## Why Detect Modality and Downstream Task?

This framework audits LLM training data quality **before** it is used for training. There are two levels of categorization, and they serve different purposes:

### Modality: fixed, determines which broad category

There are 6 modalities. They are fixed and will NOT increase:

| Modality | Directory | What kind of data |
|----------|-----------|-------------------|
| Agent Data | `modalities/Agent_Data/` | Tool calling, web navigation, code execution |
| Symbolic & Logical Data | `modalities/Symbolic_and_Logical_Data/` | Math reasoning, formal logic, code |
| Vision-Language Data | `modalities/Vision_Language_Data/` | Image-text, video-text |
| Multimodal Data | `modalities/Multimodal_Data/` | Image-to-report, cross-modal generation |
| Text Data | `modalities/Text_Data/` | Pure text corpora |
| Tabular & Graph Data | `modalities/Tabular_Data/`, `modalities/Semi_Structured_Graph_Data/` | Tables, knowledge graphs |

The modality determines the broad data-point level quality dimensions. For example, Agent data needs action validity checks, Math data needs code execution, Vision-Language data needs image existence and alignment checks. You pick from these 6 -- you never create a new modality.

### Downstream task: grows over time, determines specific metrics

Within each modality, there are **downstream tasks**. These DO grow as new datasets are added. Each downstream task has its own `data_types.py`, `loaders.py`, `executor/`, and `metrics/`, because different tasks within the same modality need different specialized evaluation:

| Modality | Downstream task | Why it needs its own metrics | Example metric |
|----------|----------------|------------------------------|----------------|
| Agent Data | **api_agent_eval** | API calls must be executable against real tool definitions | `static_executability`: validates API names, required params, param types |
| Agent Data | **text_gui_agent_eval** | Web actions must ground to correct HTML elements | `html_retention`: checks if target element is locatable in HTML; `trajectory_validity`: judges action sequence coherence |
| Symbolic & Logical | **math_eval** | Solutions must execute and produce correct answers | `validity`: runs code in sandbox, compares output to ground truth |
| Multimodal | **image_to_report_eval** | Generated reports must be accurate and well-structured | `report_quality`: evaluates clinical accuracy and hallucination |

### Metrics also live at two levels

Each modality has **Layer 1 metrics** (modality-level, universal), and each downstream task has **Layer 2 metrics** (task-specific). They are in different directories:

```
modalities/
  Agent_Data/                              # modality (fixed)
    metrics/                               # Layer 1: modality-level metrics
      diversity.py                         #   applies to ALL agent data
      fidelity.py
      safety.py
      validity.py
    api_agent_eval/                        # downstream task
      metrics/                             # Layer 2: task-specific metrics
        format_check.py                    #   specific to API agent
        executability.py                   #   (static API call validation)
        dynamic_executability.py
        diversity.py
        trustworthy.py
      data_types.py
      loaders.py
      executor/
    text_gui_agent_eval/                   # downstream task
      metrics/                             # Layer 2: task-specific metrics
        format_check.py                    #   specific to GUI agent
        static_executability.py
        html_retention.py
        trajectory_validity.py
        task_complexity.py
      data_types.py
      loaders.py
      executor/
  Symbolic_and_Logical_Data/               # modality (fixed)
    metrics/                               # Layer 1
      faithfulness.py
      fidelity.py
      validity.py
    math_eval/                             # downstream task
      metrics/                             # Layer 2
        format_check.py
        validity.py                        #   (code execution + answer check)
        faithfulness.py
        reasoning_validity.py
```

When analyzing a new dataset, you should read both levels of metrics to judge:
- Which **Layer 1 metrics** from the modality already apply? (Usually all of them do.)
- Which **Layer 2 metrics** from an existing downstream task can be reused? Which cannot?
- Does the new dataset require metrics that don't exist yet?

For example, if you see a new agent dataset that involves browser actions but in a different format than Mind2Web, most `text_gui_agent_eval/metrics/` should still apply -- but maybe `html_retention` does not if the data has no raw HTML. This kind of judgment is what you need to report in Step 5.

### What you need to determine

1. **Modality**: pick one of the 6 existing modalities (never create new)
2. **Downstream task**: does it match an existing task, or does it need a new `*_eval/` folder?
3. **Metrics compatibility**: which existing Layer 1 and Layer 2 metrics apply, which don't, and what might be missing?

If the dataset fits an existing downstream task (e.g., a new web navigation dataset that fits `text_gui_agent_eval`), we reuse the existing data_types and just add a new loader + dataset-specific executor. If it represents a genuinely new task within the modality (e.g., a code agent dataset under Agent Data), we create a new `*_eval/` directory with its own data_types, loaders, and metrics.

## Step 1: Read the Data

Do NOT skim. Read thoroughly.

1. Check file format and size:
   ```bash
   ls -la <path>
   wc -l <path>           # for jsonl
   head -c 200 <path>     # peek at first bytes
   ```

2. Read entries from multiple positions. For a 1000-line jsonl, read lines 1-5, 500-505, 995-1000. For a json array, stream-read first and last entries. The goal is to see both typical and edge-case entries.

3. For every entry, look at ALL fields including nested ones. If a field contains a list of dicts, read into that list. If there are HTML strings, note their structure. If there are conversation turns, count them.

4. Note these specifically:
   - Total number of entries
   - All top-level field names
   - Nested field structure (especially lists of dicts)
   - String field lengths (short labels vs long text vs huge HTML)
   - Whether entries are single-turn or multi-turn
   - Whether there are images/videos/code/HTML/tool definitions

## Step 2: Read the Framework Code

Read the following files. All paths are relative to project root.

**Downstream task data_types** (to understand what existing tasks expect):
- `/modalities/Agent_Data/api_agent_eval/data_types.py` -- APIAgentSample
- `/modalities/Agent_Data/text_gui_agent_eval/data_types.py` -- Record, Action
- `/modalities/Symbolic_and_Logical_Data/math_eval/data_types.py` -- MathSample
- `/modalities/Multimodal_Data/image_to_report_eval/data_types.py` -- ImageToReportSample

**Modality-level metrics** (Layer 1, to assess what universal checks exist):
- `/modalities/Agent_Data/metrics/` -- diversity.py, fidelity.py, safety.py, validity.py
- `/modalities/Symbolic_and_Logical_Data/metrics/` -- faithfulness.py, fidelity.py, validity.py

**Task-level metrics** (Layer 2, to assess what task-specific checks exist):
- `/modalities/Agent_Data/api_agent_eval/metrics/` -- format_check, executability, etc.
- `/modalities/Agent_Data/text_gui_agent_eval/metrics/` -- static_executability, html_retention, etc.
- `/modalities/Symbolic_and_Logical_Data/math_eval/metrics/` -- validity, faithfulness, etc.

**Top-level config**:
- `/evaluate.py` lines 54-91 -- the MODALITIES dispatch config

Quick reference for matching (see the "Why" section above for full details):

| Modality | data_types | Signature fields | Existing datasets |
|----------|------------|-----------------|-------------------|
| API Agent | APIAgentSample | query, tools, api_calls | toolbench, xlam, arcee |
| GUI Agent | Record(Action) | actions, cleaned_html, action_type | mind2web, webshop, weblinx |
| Math | MathSample | question, solution, ground_truth | lila, openmathinstruct |
| Image-to-Report | ImageToReportSample | instruction, report, images | iu_xray, sharegpt4v |

## Step 3: Match Modality & Downstream Task

Two decisions to make, in order:

### 3a. Pick the modality (from 6, never create new)

Based on the data content, pick one of: Agent Data, Symbolic & Logical, Vision-Language, Multimodal, Text, Tabular & Graph. This should be straightforward from the fields.

### 3b. Match the downstream task (may need to create new)

Within the matched modality, decide which case applies:

- **Case A: Existing task, existing dataset.** The data is clearly from an already-registered dataset (e.g., user gives you ToolBench data). -> Skip straight to audit.
- **Case B: Existing task, new dataset.** The data fits an existing downstream task but is a new dataset (e.g., a new web navigation dataset that fits `text_gui_agent_eval`). -> Reuse data_types, need new loader + dataset-specific executor.
- **Case C: New downstream task.** The data belongs to this modality but does not fit any existing downstream task (e.g., a code agent dataset under Agent Data, which is neither API calling nor GUI navigation). -> Need new `*_eval/` folder with data_types, loaders, executor, and metrics.

### 3c. Assess existing metrics compatibility

You already read the metrics code in Step 2. Now evaluate which ones apply to this new dataset:

1. **Layer 1 metrics** (from `modalities/<matched_modality>/metrics/`): go through each one. These usually all apply since they are modality-universal, but confirm.
2. **Layer 2 metrics** (from `modalities/<matched_modality>/<matched_task>/metrics/`):
   - If Case B: go through each metric and judge whether it makes sense for the new dataset. For example, `html_retention` only applies if the data contains HTML.
   - If Case C: read metrics from the most similar existing task as reference. Note which ones could be adapted and which are irrelevant.

This step focuses on **what already exists and whether it fits**. Step 4 will then think about **what should exist but doesn't yet**.

Explain your reasoning for each decision. What fields made you decide? What is ambiguous?

## Step 4: Find Similar Datasets & Inform data_types Design

Use WebSearch to find 2-3 similar open-source datasets for the same downstream task. Search for:
- The dataset name if identifiable
- The task type + "dataset" + "huggingface"
- Specific field patterns you see in the data

**Why this matters**: the goal is to design a `data_types.py` that is **universal for this downstream task**, not just for this one dataset. You can only get that right by seeing how multiple datasets for the same task are structured.

For example, look at how the existing API Agent data_types was designed (see `/modalities/Agent_Data/api_agent_eval/data_types.py`):

- **Fixed fields** (`query`, `tools`, `api_calls`, `final_answer`) capture what ALL API agent datasets share -- the user query, available tools, the calls made, and the final response. These are the essential fields that define this downstream task.
- **`metadata` dict** holds dataset-specific fields that other datasets in the same task may not have. For example, ToolBench has a `thought` field for chain-of-thought reasoning. This field is not universal to all API agent datasets, but it is useful for ToolBench-specific metrics (e.g., checking thought length in format_check). So it goes into `metadata`.

When you find similar datasets, compare their structures:
- What fields do ALL of them share? -> These become the fixed fields in data_types
- What fields are unique to some datasets? -> These go into metadata
- Are there fields that seem important for evaluation but only exist in certain datasets? Note these -- they may drive dataset-specific metrics later

Also think about what these datasets are used for in post-training (SFT, RLHF, DPO, etc.), because this informs what task-specific metrics should exist:
- If datasets are used for SFT on action sequences, then action correctness and trajectory coherence metrics are essential
- If the task involves calling real APIs, then executability metrics are critical
- If datasets include reward signals or human preferences, then reward distribution or preference consistency metrics might be needed
- If dataset-specific fields (like ToolBench's `thought`) enable extra quality checks, note this -- they motivate dataset-specific metrics that use metadata fields

Report your field analysis in Step 5 with a clear split: "universal fields for this task" vs "dataset-specific fields for metadata", and your metric reasoning.

## Step 5: Report to User

Present a clear summary covering all of the following:

1. **Dataset overview**: file format, total entries, what a typical entry looks like, key fields and nesting
2. **Modality**: which of the 6, and why
3. **Downstream task**: Case A/B/C, which existing task it matches (or what new task it needs), and why
4. **Similar datasets found**: names, sources, how they relate to this dataset
5. **Field mapping proposal**: which user fields map to which data_types fields, what goes in metadata
6. **Metrics compatibility**:
   - Layer 1 (modality-level): list each metric, whether it applies
   - Layer 2 (task-level): if matching an existing task, list each metric and whether it applies to this new dataset; if new task, what metrics might be needed
   - Any gaps: metrics that should exist but don't yet
7. **Open questions**: anything ambiguous, ask the user

Then wait. Do NOT proceed to code generation until the user confirms.

## What NOT to Do

- Do not write any Python code in this round
- Do not create any files
- Do not guess field meanings without reading actual data
- Do not skip reading the framework's existing data_types
- Do not proceed to Round 2 without user confirmation
