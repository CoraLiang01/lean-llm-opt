# The original model-specific functions above stay intact; these controls scope each fold.
LOTO_STATE = None
_LOTO_BASE_CLASSIFIER_PREFIX = getattr(globals().get("_loto_classifier_prefix"), "base_prefix", prefix)
_LOTO_BASE_CASE_FINGERPRINT = getattr(case_fingerprint, "_loto_base", case_fingerprint)
_LOTO_BASE_INVOKE_FORMULATION = getattr(invoke_formulation, "_loto_base", invoke_formulation)
_LOTO_BASE_EXECUTE_PIPELINE = getattr(execute_pipeline_case, "_loto_base", execute_pipeline_case)
_LOTO_BASE_FINALIZE_RECORD = getattr(finalize_record, "_loto_base", finalize_record)


class DisabledRouteError(ValueError):
    """A classifier selected a workflow that is unavailable in the current fold."""


def require_loto_fold():
    if LOTO_STATE is None:
        raise RuntimeError("Select a leave-one-type-out fold before running inference")
    return LOTO_STATE


def loto_allowed_labels():
    return list(require_loto_fold()["allowed_labels"])


def loto_reference_frame():
    """Return the filtered rows before Mixture/Others are merged into a workflow route."""
    return require_loto_fold()["reference_frame"].copy()


def _loto_classifier_prefix():
    state = require_loto_fold()
    blocks = re.split(r"(?=Example \d+)", few_shot_example)
    kept = [block for block in blocks if not re.search(
        r"Final Answer:\s*" + re.escape(state["held_out_type"]) + r"\b", block)]
    fold_prefix = _LOTO_BASE_CLASSIFIER_PREFIX.replace(few_shot_example, "".join(kept))
    if not state["disabled_route"]:
        return fold_prefix
    # Keep original semantic definitions and hand-written examples. Only availability changes.
    tokens = ", ".join(state["allowed_labels"])
    return fold_prefix + f"""

WORKFLOW AVAILABILITY FOR THIS RUN (overrides the earlier output-label list):
The only available Final Answer labels are: {tokens}.
Some examples and definitions above describe unavailable workflows. They remain
reference material, but their labels are not valid choices for this run.
Select the most appropriate AVAILABLE workflow from the current problem's structure.
Do not output any other label. Do not infer the current problem's true label from
which workflows are available. Keep the same one-FileQA-call ReAct protocol.
"""


_loto_classifier_prefix.base_prefix = _LOTO_BASE_CLASSIFIER_PREFIX


def set_loto_fold(held_out_type):
    global LOTO_STATE, prefix
    semantic_type = normalize_problem_class(held_out_type)
    if semantic_type not in CLASS_LABELS:
        raise ValueError(f"Unsupported held-out semantic type: {held_out_type!r}")
    path = Path(RAG_EXAMPLES_ALL_PATH).expanduser()
    path = path if path.is_absolute() else PROJECT_ROOT / path
    frame = read_csv_compat(path, dtype=str, keep_default_na=False)
    required = {"prompt", "Data_address", "Related", "Required Data", "Label", "Code", "Type"}
    if required.difference(frame.columns):
        raise ValueError(f"RAG example CSV missing fields: {sorted(required.difference(frame.columns))}")
    semantic_labels = frame["Type"].map(normalize_problem_class)
    remove = semantic_labels.eq(semantic_type)
    filtered = frame.loc[~remove].copy()
    if len(filtered) < 5:
        raise ValueError("At least five remaining examples are required by the original FileQA protocol")
    disabled_route = class_to_workflow_route(semantic_type) if LOTO_REMOVE_ROUTE else None
    allowed = [label for label in CLASS_LABELS if class_to_workflow_route(label) != disabled_route]
    LOTO_STATE = {
        "held_out_type": semantic_type, "disabled_route": disabled_route,
        "allowed_labels": allowed, "reference_frame": filtered,
        "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reference_path": str(path.resolve()),
        "removed_example_indices": [int(index) for index in frame.index[remove]],
        "removed_examples": [
            {"row_index": int(index), "type": str(frame.loc[index, "Type"]),
             "prompt": str(frame.loc[index, "prompt"])} for index in frame.index[remove]
        ],
        "remaining_example_indices": [int(index) for index in filtered.index],
        "remaining_counts": filtered["Type"].map(normalize_problem_class).value_counts().to_dict(),
        "reference_count_before": len(frame), "reference_count_after": len(filtered),
    }
    # Reset every reference/agent cache at fold boundaries, including the shared Others store.
    for name in ("_get_classifier_retriever", "_get_classification_agent", "_load_rag_table",
                 "_get_rag_store", "_unified_rag_rows", "get_route_retriever", "get_others_store"):
        clear = getattr(globals().get(name), "cache_clear", None)
        if clear is not None:
            clear()
    prefix = _loto_classifier_prefix()
    return loto_manifest()


def loto_manifest():
    state = require_loto_fold()
    return {
        "experiment": LOTO_VARIANT, "model": MODEL_SNAPSHOT,
        "base_notebook": LOTO_BASE_NOTEBOOK, "base_notebook_sha256": LOTO_BASE_SHA256,
        "held_out_type": state["held_out_type"], "disabled_workflow_route": state["disabled_route"],
        "allowed_labels": state["allowed_labels"],
        "allowed_routes": [route for route in WORKFLOW_ROUTES if route != state["disabled_route"]],
        "reference_path": state["reference_path"], "reference_sha256": state["reference_sha256"],
        "reference_count_before": state["reference_count_before"],
        "reference_count_after": state["reference_count_after"],
        "removed_examples": state["removed_examples"],
        "remaining_example_indices": state["remaining_example_indices"],
        "remaining_counts": state["remaining_counts"],
        "removal_scope": "Exact semantic Type in RAG_Examples_All; UFLP normalized to FLP",
        "fixed_prompt_examples": "Target-type classifier demonstrations removed; generic definitions retained",
        "classification": "Recomputed using the current fold; no external classification cache",
    }


def loto_record_fields():
    state = require_loto_fold()
    return {"loto_variant": LOTO_VARIANT, "held_out_type": state["held_out_type"],
            "disabled_workflow_route": state["disabled_route"],
            "allowed_labels": ",".join(state["allowed_labels"]),
            "reference_sha256": state["reference_sha256"],
            "reference_count_after": state["reference_count_after"]}


def assert_loto_route_allowed(route):
    state = require_loto_fold()
    route = normalize_route(route)
    if route == state["disabled_route"]:
        raise DisabledRouteError(f"Workflow {route} is disabled in the {state['held_out_type']} fold")
    return route


def invoke_formulation(route, query, dataset_address):
    # Explicit guard prevents .get(..., get_Others_response) from bypassing a disabled route.
    assert_loto_route_allowed(route if dataset_address else "Others")
    return _LOTO_BASE_INVOKE_FORMULATION(route, query, dataset_address)


invoke_formulation._loto_base = _LOTO_BASE_INVOKE_FORMULATION


def execute_pipeline_case(case, *, forced_route=None):
    require_loto_fold()
    if forced_route is not None:
        raise ValueError("LOTO inference must reclassify; forced routes are not supported")
    try:
        record = _LOTO_BASE_EXECUTE_PIPELINE(case, forced_route=None)
    except Exception as exc:
        context = {**getattr(exc, "pipeline_context", {}), **loto_record_fields()}
        if isinstance(exc, DisabledRouteError):
            context.update(pipeline_stage="route_selection", route_allowed=False)
        exc.pipeline_context = context
        raise
    record.update(**loto_record_fields(), route_allowed=True)
    return record


execute_pipeline_case._loto_base = _LOTO_BASE_EXECUTE_PIPELINE


def finalize_record(record, *, rel_tol=1e-4, abs_tol=1e-4):
    result = _LOTO_BASE_FINALIZE_RECORD(record, rel_tol=rel_tol, abs_tol=abs_tol)
    # Gold-label accuracy remains measurable; the forbidden label is expected to score zero.
    return result


finalize_record._loto_base = _LOTO_BASE_FINALIZE_RECORD


def case_fingerprint(case, route=None, *, source_fingerprint=None):
    baseline = _LOTO_BASE_CASE_FINGERPRINT(case, route, source_fingerprint=source_fingerprint)
    identity = {**loto_record_fields(), "baseline": baseline,
                "model": MODEL_SNAPSHOT}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


case_fingerprint._loto_base = _LOTO_BASE_CASE_FINGERPRINT

_LOTO_BASE_CLASSIFY = invoke_classifier
_LOTO_BASE_FORMULATE_CSVQA = formulate_with_csvqa
_LOTO_BASE_GENERATE_CODE = _generate_code


def invoke_classifier(query):
    result = _LOTO_BASE_CLASSIFY(query)
    if result["normalized_label"] not in loto_allowed_labels():
        raise DisabledRouteError("Classifier emitted an unavailable label; no alternative route is assigned")
    assert_loto_route_allowed(class_to_workflow_route(result["normalized_label"]))
    return result


def formulate_with_csvqa(query, dataset_address, route, system_prompt, tool_description, prefix, suffix):
    assert_loto_route_allowed(route)
    prefix += ("\nThe selected workflow is a source of reference patterns only. "
        "Infer the current mathematical structure entirely from the user's question and current CSV data. "
        "Do not force the question into the selected route's archetype or copy an example's extra constraints. "
        "Retain every question-specific decision family, coupling, domain and objective term, "
        "even when its structure differs from the remaining reference examples.")
    return _LOTO_BASE_FORMULATE_CSVQA(query, dataset_address, route, system_prompt, tool_description, prefix, suffix)


def _generate_code(output, route, original_query="", data_payload="", legacy_observation=""):
    assert_loto_route_allowed(route)
    return _LOTO_BASE_GENERATE_CODE(output, route, original_query, data_payload, legacy_observation)
