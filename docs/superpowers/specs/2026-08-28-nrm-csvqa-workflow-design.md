# NRM CSVQA, ReAct, and `get_code` Workflow Design

## Objective

Improve the NRM route in `LEAN_LLM_OPT_4.1_Large-scale-or_original.ipynb` while preserving the paper's architecture:

```text
classification -> type-specific NRM ReAct/CSVQA -> get_code(NRM) -> Gurobi
```

The implementation must use one validated data selection throughout formulation, code generation, validation, repair, and execution. It must not independently infer the selected rows or columns in later stages.

## Scope

The first implementation targets the benchmark's canonical NRM structure and supports:

- one CSV file;
- multiple same-schema CSV files appended in file order;
- selection of relevant files from multiple supplied files;
- explicit one-to-one or many-to-one relations between two heterogeneous tables;
- all-row and filtered-row selection;
- query-relevant column selection;
- compact observations for large selections;
- one in-agent CSVQA correction and one outer formulation retry;
- deterministic data loading in the NRM `get_code` path;
- 25 true-NRM and 76 forced-NRM evaluation cells.

The implementation must reject ambiguous joins, unsupported many-to-many relationships, and unresolved required roles instead of silently using all data.

## Authority Boundaries

Three artifacts have non-overlapping authority:

- The original query is authoritative for business meaning, special constraints, and variable domains.
- The validated selection manifest is authoritative for files, source-row positions, selected columns, table relations, and parameter-column bindings.
- The ReAct formulation is authoritative for the objective and constraint construction logic.

If these artifacts conflict, the workflow must fail or repair the formulation. `get_code` must not resolve the conflict by changing the data selection.

## End-to-End Data Flow

```text
query + dataset paths
    -> deterministic schema inspection
    -> concise schema card
    -> NRM Ref-Data example retrieval
    -> NRM ReAct agent
    -> compact JSON CSVQA action input
    -> deterministic plan validation and extraction
    -> immutable selection manifest + observation
    -> ReAct mathematical formulation
    -> formulation/manifest consistency checks
    -> deterministic loader generated from the manifest
    -> LLM-generated Gurobi model body based on the formulation
    -> static validation
    -> isolated execution
    -> one model-body repair when eligible
```

## Schema Inspection

### Responsibilities

The schema inspector reads only factual metadata before ReAct calls CSVQA:

- short file IDs such as `F1` and `F2`;
- exact column names;
- source row count;
- inferred column kind: string, numeric, date-like, Boolean, or unknown;
- null and unique-value counts;
- at most three JSON-escaped, length-limited non-null samples;
- conservative NRM role suggestions with confidence.

Samples are untrusted data. Prompt instructions must state that sample contents can be used only as evidence about column meaning and must never be followed as instructions.

### Role Suggestions

Role suggestions are evidence, not automatic bindings:

- High confidence: an exact normalized match to a specific alias.
- Medium confidence: a specific multi-token or substring match.
- Unresolved: generic or anonymous names such as `value`, `quantity`, `amount`, `c1`, or `q7` without sufficient query evidence.

Ambiguous names must not be force-mapped. ReAct chooses exact columns after seeing the query, schema card, and suggestions.

### Exported Index Columns

A column such as `Unnamed: 0` may be removed only if all conditions hold:

- its name indicates an exported index;
- its values form an index-like sequence;
- the query does not reference it;
- it is not selected as an identifier, filter, role, relation key, or extra model column.

A product-name column must never be removed solely because its header resembles an exported index.

## Compact CSVQA Action Input

ReAct writes a short JSON selection intent. It does not write paths, source positions, row counts, hashes, excluded columns, or data values.

Single-file all-row example:

```json
{
  "file": "F1",
  "rows": {"mode": "all"},
  "identifier": ["Product_Code"],
  "roles": {
    "revenue": "Unit_Value",
    "demand": "Request_Qty",
    "inventory": "Available_Stock"
  },
  "extra": []
}
```

Filtered example:

```json
{
  "file": "F1",
  "rows": {
    "mode": "filter",
    "logic": "all",
    "where": [["Product_Code", "starts_with", "X"]]
  },
  "identifier": ["Product_Code"],
  "roles": {
    "revenue": "Unit_Value",
    "demand": "Request_Qty",
    "inventory": "Available_Stock"
  },
  "extra": []
}
```

Multiple identical-schema files use `files` plus `combine: "append"`. Heterogeneous files use per-file plans and explicit validated relations.

## Stateful CSVQA Session

Each problem creates an isolated session object. No global manifest registry is permitted. The session stores:

- inspected schemas;
- call count;
- validation errors;
- at most one successful manifest;
- the observation created from that manifest.

The first successful plan locks the session. A second call is allowed only after a failed call. The normal path therefore contains one CSVQA call; the corrected path contains one failed call and one successful call.

## Selection Validation

The deterministic validator must check:

- valid JSON and allowed fields;
- existing file IDs and exact column names;
- `all` or `filter` row mode;
- no filters in `all` mode;
- at least one filter in `filter` mode;
- allowed operators and type-compatible operands;
- query-supported filter values and no invented subset;
- required NRM role bindings;
- distinct and numeric-convertible core role columns;
- required extra columns implied by the query;
- non-empty selection;
- no missing or non-numeric values in required model roles;
- preserved source order;
- valid join keys and cardinality for multi-table plans.

Allowed filter operators are:

- strings: `equals`, `starts_with`, `contains`, `in`;
- numeric/date-like columns: `eq`, `gt`, `ge`, `lt`, `le`, `between`, `in`.

Arbitrary Python expressions, regular-expression code, and `eval` are forbidden.

Invalid plans return a short structured `CSVQA_ERROR` containing an error code, the invalid field, available choices, and a retry instruction. They do not return full data.

## Deterministic Extraction

The extractor must:

- remove only fully empty rows and validated exported-index columns;
- preserve file order and source-row order;
- never sort, deduplicate, sample, or reset indices;
- store original zero-based row positions;
- select columns by positive inclusion rules;
- validate parameter types before returning success.

Column sets are derived as follows:

```text
model columns = role columns + extra model columns
observation columns = identifiers + model columns + filter audit columns
runtime columns = columns required to construct the executable model
```

Irrelevant date, comment, and metadata columns are omitted unless they are used for filtering, relations, model parameters, or query-defined constraints.

## Selection Manifest

Python generates a JSON-serializable manifest containing:

- an integrity-only `selection_id`;
- query hash;
- exact resolved paths and file-content hashes;
- source schemas and row counts;
- selected original row positions for every file;
- exact role, identifier, extra, observation, and runtime columns;
- validated relations and cardinalities;
- expected selected row counts;
- cleanup and integrity audit fields;
- observation mode.

The manifest is passed by value inside the workflow result and persisted in checkpoints. `selection_id` is not a lookup key and is not required in the paper-facing formulation.

## Observation Modes

### Materialized

For at most 30 selected rows, return:

- selection summary;
- exact role and column bindings;
- each selected row as escaped JSON;
- dense `variable_index` and original source position.

### Manifest-Only

For more than 30 selected rows, return:

- file IDs;
- selected row count;
- row mode and filter summary;
- exact role and column bindings;
- validation status;
- an instruction to construct a generic indexed model.

Do not enumerate parameter arrays. If the query contains group, time, shared-capacity, or relation-dependent structure, return the necessary compact sets and mappings even in manifest-only mode. Row count controls value expansion; model structure controls required metadata.

## ReAct Formulation Contract

The ReAct final answer must be produced only after a successful CSVQA observation and must contain:

```text
Data Scope
Sets
Parameters and Column Bindings
Decision Variables
Objective
Constraints
Variable Domains
```

The formulation must:

- use only selected files, rows, and columns;
- preserve the validated ordered-record interpretation;
- avoid inventing or copying long arrays in manifest-only mode;
- derive special constraints from the original query;
- avoid introducing new filters after CSVQA success;
- use query-specified variable domains, with the paper's NRM default only when unspecified.

The wrapper checks for exactly one successful CSVQA call, no more than one failed call before it, a valid manifest, and a formulation consistent with the available bindings.

## Workflow Result

`get_NRM_response` returns a dictionary rather than only a string:

```python
{
    "status": "completed",
    "formulation": formulation,
    "selection_manifest": manifest,
    "observation_summary": observation,
    "workflow_trace": intermediate_steps,
    "csvqa_calls": call_count,
    "workflow_attempts": attempt_count,
}
```

The NRM routing caller unwraps this object. Other route handlers remain unchanged.

## ReAct Repair Policy

Within one ReAct run, CSVQA permits one invalid call followed by one corrected call. If no valid formulation and manifest are produced, the outer NRM workflow runs once more with concise prior diagnostics and a new isolated session. If the second workflow attempt fails, the instance stops with `formulation_data_selection_failed`. Explicit filtered requests never silently fall back to all rows.

## NRM `get_code` Handoff

`get_code` accepts the exact manifest:

```python
get_code(
    formulation=workflow_result["formulation"],
    selected_problem="NRM",
    selection_manifest=workflow_result["selection_manifest"],
    original_query=query,
)
```

The original query is audit context only and cannot change the selected data.

### Deterministic Loader

Python generates the NRM data-loading preamble from the manifest. It:

- verifies file hashes and schemas;
- reads exact runtime columns;
- selects exact original row positions with `iloc`;
- preserves their stored order;
- converts required role columns with `errors="raise"`;
- asserts every role array length equals the expected selected row count.

The loader exposes stable bindings such as:

```python
selected_data
role_data["revenue"]
role_data["demand"]
role_data["inventory"]
n
```

### LLM-Generated Model Body

The NRM code-generation prompt receives the formulation and stable bindings. The LLM generates only the Gurobi model body: variables, objective, constraints, domains, and `optimize()`. It must not read files, select rows, hard-code long arrays, or overwrite reserved loader bindings.

The final program is assembled from fixed imports, the deterministic loader, and the generated model body. NRM code examples must use `role_data` rather than literal parameter arrays.

## Static Validation and Code Repair

Static validation receives the same manifest by value and never rebuilds a runtime specification from the query. It checks:

- immutable loader/path/row/column bindings;
- unchanged file fingerprints;
- role-array lengths;
- no second CSV read or data reselection;
- no sorting, deduplication, sampling, or index reset;
- no hard-coded NRM parameter arrays;
- objective direction and variable domain consistency;
- required constraint families;
- `optimize()` invocation;
- preservation of the solved model object.

One eligible repair may modify only the Gurobi model body. The manifest and deterministic loader remain unchanged.

## Notebook Integration

The implementation changes only the NRM-specific path and shared functions that must accept the manifest. It preserves classification, other typed handlers, Ref-Data retrieval, ReAct, CSVQA, type-specific `get_code`, execution isolation, and routing evaluation.

Code-cell requirements:

- no `v1`, `v2`, or `v3` suffixes;
- no Chinese code or inline code comments;
- standard English docstrings stating inputs, outputs, and purpose;
- Chinese Markdown cells documenting changed sections and rationale;
- no unrelated notebook refactoring.

## Evaluation Cells

The notebook must include disabled-by-default cells for two resumable evaluations using the same correctness function and output schema.

### True-NRM Evaluation

- Select all 25 benchmark rows whose true label is NRM.
- Force route `NRM`.
- Run the complete NRM ReAct/CSVQA/manifest/`get_code` workflow.
- Save checkpoint JSONL and summary CSV.
- Report correct count, accuracy, execution failures, selection failures, repairs, and per-row diagnostics.

### Non-NRM Forced-NRM Evaluation

- Select the remaining 76 benchmark rows.
- Force route `NRM`.
- Reuse the exact same workflow and correctness rule.
- Save separate checkpoint JSONL and summary CSV.
- Summarize correct/total/accuracy by true category.

The two runs must not share result files with older protocol fingerprints.

## Verification Sequence

1. Deterministic unit checks without LLM calls:
   - all rows with irrelevant date columns;
   - string-prefix and exact filters;
   - numeric and date ranges;
   - true exported index versus product-valued `Unnamed: 0`;
   - unknown columns and ambiguous roles;
   - empty selections;
   - 30/31-row observation boundary;
   - same-schema append;
   - validated one-to-one and many-to-one relations;
   - manifest-loader equality.
2. Parse every notebook code cell.
3. Confirm no prohibited suffixes or Chinese text in code cells.
4. Run a five-instance NRM pilot.
5. Run all 25 true-NRM instances.
6. Run the remaining 76 instances forced to NRM.
7. Compare objective accuracy, failures, repair rate, token use, and latency with the previous NRM results.

## Acceptance Criteria

- One validated manifest is reused by formulation, code generation, validation, repair, and execution.
- No NRM stage independently rebuilds row or column selection.
- Explicit subset failures never become silent all-row runs.
- Large selections do not produce copied parameter arrays.
- The deterministic loader exactly reproduces manifest source positions and runtime columns.
- The 25-row and 76-row evaluation cells are directly runnable and resumable.
- Other route handlers remain behaviorally unchanged.
