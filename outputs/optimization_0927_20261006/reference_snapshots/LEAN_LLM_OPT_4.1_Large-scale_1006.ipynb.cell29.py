def _source_candidate(code):
    source = str(code).strip()
    match = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", source, re.DOTALL)
    source = match.group(1) if match else source
    tree = ast.parse(source)
    # Bulk names expand business keys; keep those keys out of Gurobi display names.
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"addVars", "addConstrs"}):
            node.keywords = [kw for kw in node.keywords if kw.arg != "name"]
            node.keywords.append(ast.keyword(arg="name", value=ast.Constant(value="")))
    return ast.unparse(tree)


class ModelInterfaceError(RuntimeError):
    """Generated program did not expose a live Gurobi model."""

    def __init__(self, code, detail):
        self.code = code
        super().__init__(f"{code}: {detail}")


def _resolve_generated_model(namespace):
    """Prefer m/model, otherwise accept one distinct global model; never rerun code."""
    for name in ("m", "model"):
        candidate = namespace.get(name)
        if isinstance(candidate, gp.Model):
            return candidate
    candidates = {}
    for name, value in namespace.items():
        if isinstance(value, gp.Model):
            candidates.setdefault(id(value), (value, []))[1].append(name)
    if not candidates:
        types = {name: type(namespace[name]).__name__ for name in ("m", "model") if name in namespace}
        raise ModelInterfaceError(
            "MODEL_INTERFACE_MISSING",
            f"No global Gurobi model was exposed; m/model types: {types}. "
            "A function-local model must be explicitly returned and assigned to m.",
        )
    if len(candidates) != 1:
        names = [aliases for _, aliases in candidates.values()]
        raise ModelInterfaceError("MODEL_INTERFACE_AMBIGUOUS", f"Multiple distinct global Gurobi models: {names}")
    return next(iter(candidates.values()))[0]


def execute_code(code):
    namespace = {"__name__": "__main__"}
    exec(_source_candidate(code), namespace)
    model = _resolve_generated_model(namespace)
    try:
        status = model.Status
    except (gp.GurobiError, AttributeError) as exc:
        raise ModelInterfaceError(
            "MODEL_INTERFACE_DISPOSED", "The exposed model is unavailable; do not dispose it before result extraction."
        ) from exc
    with model:
        if status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"No optimal solution; solver status: {status}")
        objective = model.ObjVal
        solution = [(v.VarName, v.X) for v in model.getVars()]
        print("Optimal objective:", objective)
        print("Optimal solution:", solution)
        return objective, solution


ROUTE_FORMULATORS = {
    "NRM": get_NRM_response, "RA": get_RA_response, "TP": get_TP_response,
    "AP": get_AP_response, "FLP": get_FLP_response,
}


def invoke_formulation(route, query, dataset_address):
    if not dataset_address:
        return {"formulation": get_others_without_CSV_response(query), "trace": {"status": "NOT_APPLICABLE"}}
    return ROUTE_FORMULATORS.get(normalize_route(route), get_Others_response)(query, dataset_address)


def case_fields(case):
    reference = case.get("label_objective")
    address = normalize_data_address(case.get("dataset_address"))
    return {
        "problem_id": str(case["problem_id"]), "query": str(case["Query"]),
        "dataset_address": address, "input_kind": "external_csv" if address else "query_only",
        "true_label": case.get("true_label"), "true_route": case.get("true_route"),
        "label_objective": float(reference) if reference is not None and pd.notna(reference) else None,
    }


def classification_for_case(case):
    return invoke_classifier(case["Query"])


def execute_pipeline_case(case, *, forced_route=None):
    """Classify, formulate, generate code, and solve; retain progress if a step fails."""
    record = {**case_fields(case), "forced_route": forced_route,
              "experiment_mode": "forced" if forced_route else "automatic",
              "repair_count": 0, "retry_count": 0, "fallback_count": 0}
    global REACT_PROTOCOL_DEADLINE
    import time
    REACT_PROTOCOL_EVENTS.clear()
    REACT_PROTOCOL_DEADLINE = time.monotonic() + REACT_PROTOCOL_TIMEOUT_SECONDS
    HTTP_RETRY_EVENTS.clear()
    query, address = record["query"], record["dataset_address"]
    try:
        if forced_route is not None and not address:
            raise ValueError("Forced routes require CSV inputs")
        record["pipeline_stage"] = "classification" if forced_route is None else "formulation"
        classification = classification_for_case(case) if forced_route is None else {}
        label = classification.get("normalized_label")
        route = normalize_route(forced_route) if forced_route is not None else class_to_workflow_route(label)
        record.update(predicted_label=label, assigned_route=route if address else "Others")
        record["pipeline_stage"] = "formulation"
        formulation = invoke_formulation(route, query, address)
        mode = CSVQA_MODE_BY_ROUTE[route] if address else None
        payload = formulation.get("observation", "") if mode == "planned" else ""
        record.update(generated_model=formulation["formulation"], csvqa_observation=formulation.get("observation", ""), csvqa_mode=mode,
                      csvqa_status=formulation.get("trace", {}).get("status"),
                      csvqa_trace=json.dumps(formulation.get("trace", {}), ensure_ascii=False),
                      fallback_count=formulation.get("trace", {}).get("fallback_count", 0))
        model_text = formulation["formulation"]
        if (not isinstance(model_text, str) or not model_text.strip()
                or "agent stopped due to" in model_text.lower()):
            raise RuntimeError("Invalid formulation: empty model or agent stopped before producing a model")
        record["pipeline_stage"] = "code_generation"
        code = formulation.get("code")
        if code is None:
            if address:
                code = get_csv_code(
                    formulation["formulation"], route, query, data_payload=payload,
                    legacy_observation=formulation.get("observation", "")
                    if route == "RA" and mode == "legacy" else "",
                )
            else:
                code = get_code(formulation["formulation"], route, original_query=query)
        source = _source_candidate(code)
        if payload:
            # Include NRM data so the saved program can run on its own.
            data = json.loads(payload) if isinstance(payload, str) else payload
            source = (f"CSVQA_DATA = {pprint.pformat(data, sort_dicts=False, width=120)}\n"
                      "import pandas as pd\n"
                      "CSVQA_FRAMES = {t[\"table_id\"]: pd.DataFrame("
                      "[r[\"values\"] for r in t[\"records\"]], columns=t[\"columns\"], "
                      "index=[r[\"source_row\"] for r in t[\"records\"]]) "
                      "for t in CSVQA_DATA[\"tables\"]}\n" + source)
        record["solve_code"] = source
        record["pipeline_stage"] = "solve"
        objective, solution = execute_code(record["solve_code"])
        record.update(**react_protocol_record_fields())
        record.update(api_retry_count=len(HTTP_RETRY_EVENTS), api_retry_events=json.dumps(HTTP_RETRY_EVENTS))
        record.update(final_objective=objective, final_solution=solution, final_ok=True,
                      record_status="completed", cache_source="computed", pipeline_stage="completed")
        return record
    except Exception as exc:
        record.update(**react_protocol_record_fields())
        record.update(api_retry_count=len(HTTP_RETRY_EVENTS), api_retry_events=json.dumps(HTTP_RETRY_EVENTS))
        evidence = getattr(exc, "csvqa_result", {})
        if evidence:
            record.update(csvqa_observation=evidence.get("observation", ""),
                          csvqa_trace=json.dumps(evidence.get("trace", {}), ensure_ascii=False),
                          csvqa_status=evidence.get("trace", {}).get("status"),
                          fallback_count=evidence.get("trace", {}).get("fallback_count", 0))
        record["pipeline_stage"] = getattr(exc, "pipeline_stage", record["pipeline_stage"])
        if isinstance(exc, ModelInterfaceError):
            record["pipeline_stage"] = "result_extraction"
        exc.pipeline_context = record
        raise
