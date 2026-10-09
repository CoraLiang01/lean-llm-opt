def _source_candidate(code):
    source = str(code).strip()
    match = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", source, re.DOTALL)
    return match.group(1) if match else source


def _normalize_generated_source(source, payload=None):
    """Compile shared one-pass execution contracts before the only execution."""
    import ast
    import io
    import tokenize

    replacements = {"null": "None", "true": "True", "false": "False"}
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NAME and token.string in replacements:
            token = tokenize.TokenInfo(
                token.type, replacements[token.string], token.start, token.end, token.line
            )
        tokens.append(token)
    normalized = tokenize.untokenize(tokens)

    alias_targets = {}
    ambiguous_aliases = set()
    for table in (payload or {}).get("tables", []):
        canonical = table.get("table_id")
        aliases = list(table.get("aliases") or []) + [table.get("file_name")]
        for alias in aliases:
            if not alias or not canonical:
                continue
            if alias in alias_targets and alias_targets[alias] != canonical:
                ambiguous_aliases.add(alias)
            else:
                alias_targets[alias] = canonical
    for alias in ambiguous_aliases:
        alias_targets.pop(alias, None)

    class NormalizeExecutionContract(ast.NodeTransformer):
        cosmetic_name_methods = {"addVar", "addVars", "addConstr", "addConstrs", "addQConstr"}

        def __init__(self):
            self.indexed_columns = {}

        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in alias_targets:
                return ast.copy_location(ast.Constant(alias_targets[node.value]), node)
            return node

        def visit_ImportFrom(self, node):
            if node.module == "gurobipy":
                node.names = [alias for alias in node.names if alias.name != "gp"]
                if not node.names:
                    return None
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "MIPGap":
                    node.value = ast.Constant(1e-6)
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "set_index"
                and node.value.args
                and isinstance(node.value.args[0], ast.Constant)
                and isinstance(node.value.args[0].value, str)
            ):
                self.indexed_columns.setdefault(node.targets[0].id, set()).add(node.value.args[0].value)
            return node

        def visit_AnnAssign(self, node):
            self.generic_visit(node)
            if isinstance(node.target, ast.Attribute) and node.target.attr == "MIPGap":
                node.value = ast.Constant(1e-6)
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr in self.cosmetic_name_methods:
                node.keywords = [keyword for keyword in node.keywords if keyword.arg != "name"]
            if isinstance(node.func, ast.Attribute) and node.func.attr == "setParam" and len(node.args) >= 2:
                parameter = node.args[0]
                is_gap = (
                    isinstance(parameter, ast.Constant) and parameter.value == "MIPGap"
                ) or (isinstance(parameter, ast.Attribute) and parameter.attr == "MIPGap")
                if is_gap:
                    node.args[1] = ast.Constant(1e-6)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "drop"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in self.indexed_columns
            ):
                columns_keyword = next((item for item in node.keywords if item.arg == "columns"), None)
                if columns_keyword and isinstance(columns_keyword.value, (ast.List, ast.Tuple)):
                    kept = [
                        item for item in columns_keyword.value.elts
                        if not (
                            isinstance(item, ast.Constant)
                            and item.value in self.indexed_columns[node.func.value.id]
                        )
                    ]
                    if not kept:
                        return ast.copy_location(node.func.value, node)
                    columns_keyword.value.elts = kept
            return node

    tree = NormalizeExecutionContract().visit(ast.parse(normalized))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def execute_code(code):
    gp.setParam("TimeLimit", float(os.environ.get("LEAN_GUROBI_TIME_LIMIT", "180")))
    gp.setParam("MIPGap", 1e-6)
    namespace = {
        "__name__": "__main__", "gp": gp, "GRB": gp.GRB,
        "pd": pd, "np": np, "math": math, "json": json, "Path": Path,
    }
    original_read_csv = pd.read_csv

    def clean_generated_read_csv(*args, **kwargs):
        frame = original_read_csv(*args, **kwargs)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return frame
        # Generated programs may intentionally use an index-looking column.
        # Preserve every column and remove only empty/repeated-header rows.
        nonempty_rows = frame.apply(
            lambda column: column.map(_nonempty_cell)
        ).any(axis=1)
        frame = frame.loc[nonempty_rows].copy()
        repeated_headers = _repeated_header_mask(frame)
        return frame.loc[~repeated_headers].copy() if repeated_headers.any() else frame

    # Apply the same deterministic input adapter even if generated code directly reads a CSV.
    pd.read_csv = clean_generated_read_csv
    try:
        exec(_source_candidate(code), namespace)
    finally:
        pd.read_csv = original_read_csv
    model = namespace.get("m") or namespace.get("model")
    if not isinstance(model, gp.Model):
        models = [value for value in namespace.values() if isinstance(value, gp.Model)]
        if len(models) != 1:
            raise RuntimeError(f"Generated code must expose exactly one Gurobi model; found {len(models)}")
        model = models[0]
    with model:
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"No optimal solution; solver status: {model.Status}")
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


def execute_pipeline_case(case, *, forced_route=None):
    """Classify, formulate, generate code, and solve; retain progress if a step fails."""
    record = {**case_fields(case), "forced_route": forced_route,
              "experiment_mode": "forced" if forced_route else "automatic"}
    query, address = record["query"], record["dataset_address"]
    try:
        if forced_route is not None and not address:
            raise ValueError("Forced routes require CSV inputs")
        classification = invoke_classifier(query) if forced_route is None else {}
        label = classification.get("normalized_label")
        route = normalize_route(forced_route) if forced_route is not None else class_to_workflow_route(label)
        record.update(predicted_label=label, assigned_route=route if address else "Others")
        formulation = invoke_formulation(route, query, address)
        formulation_text = str(formulation.get("formulation", "") or "").strip()
        if not formulation_text or "Agent stopped due to iteration limit or time limit" in formulation_text:
            raise RuntimeError("Formulation agent did not return a complete mathematical model")
        formulation["formulation"] = formulation_text
        mode = CSVQA_MODE_BY_ROUTE[route] if address else None
        payload = formulation.get("observation", "") if mode == "planned" else ""
        if mode == "planned":
            payload_object = json.loads(payload) if isinstance(payload, str) else payload
            validation_status = (payload_object.get("validation") or {}).get("status")
            if validation_status not in {"OK", "FALLBACK_FULL_DATA"}:
                raise RuntimeError(f"CSVQA data validation did not complete: {validation_status!r}")
        record.update(generated_model=formulation["formulation"], csvqa_observation=formulation.get("observation", ""), csvqa_mode=mode,
                      csvqa_status=formulation.get("trace", {}).get("status"))
        code = formulation.get("code")
        if code is None:
            if address:
                code = get_csv_code(formulation["formulation"], route, query, data_payload=payload)
            else:
                code = get_code(formulation["formulation"], route)
        source = _normalize_generated_source(_source_candidate(code))
        if payload:
            # Include NRM data so the saved program can run on its own.
            data = json.loads(payload) if isinstance(payload, str) else payload
            source = f"CSVQA_DATA = {pprint.pformat(data, sort_dicts=False, width=120)}\n{source}"
        record["solve_code"] = source
        objective, solution = execute_code(record["solve_code"])
        record["code_repair_count"] = 0
        record.update(final_objective=objective, final_solution=solution, final_ok=True,
                      record_status="completed", cache_source="computed")
        return record
    except Exception as exc:
        exc.pipeline_context = record
        raise
