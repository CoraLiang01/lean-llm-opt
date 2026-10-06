"""Shared source embedded in the three GPT-4.1 Model_Interface_V2 notebooks.

Uses the notebooks' existing ast/gp/hashlib/json/math/re imports. No model,
data-extraction, classification, retry, or solver-parameter changes live here.
"""

MODEL_INTERFACE_VERSION = "model-interface-v2"
MODEL_RETURN_CONTRACT = """
EXECUTION INTERFACE (applies even when no reference examples are available):
Create exactly one Gurobi Model and expose it at module scope as m.
Prefer a top-level script. If you use a solving function, explicitly return its
Gurobi Model and call the function at module scope: m = solve_problem(...).
Defining a function without calling it is not a completed program.
Never assign m = m.optimize(): optimize() does not return the model.
Reserve m/model for the model; do not overwrite them with a scalar or None.
Call optimize() exactly once. Leave the model alive for the execution harness:
do not dispose/close it or use a model context manager before results are read.
Preserve the supplied formulation, variable domains, data, and objective.
These interface rules take precedence over the style of reference code.
"""


class ModelContractError(RuntimeError):
    """The program did not expose one unambiguous, readable Gurobi model."""


class SolverStatusError(RuntimeError):
    """The generated model did not finish with the required optimal status."""


def _instrument_model_creation(source):
    """Retain model objects without altering math, invoking functions, or solving.

    Wrap only statically identified gurobipy.Model constructor calls. The saved
    executed source includes this helper and can be inspected/replayed directly.
    This is an interface adapter, not an untrusted-code sandbox.
    """
    tree = ast.parse(source)
    reserved = {"__lean_models_v2", "__lean_capture_v2"}
    if any(isinstance(node, ast.Name) and node.id in reserved for node in ast.walk(tree)):
        raise ModelContractError("Generated code uses a reserved executor identifier")
    module_names, constructor_names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_names.update(alias.asname or alias.name for alias in node.names
                                if alias.name == "gurobipy")
        elif isinstance(node, ast.ImportFrom) and node.module == "gurobipy":
            for alias in node.names:
                if alias.name == "Model":
                    constructor_names.add(alias.asname or alias.name)
                elif alias.name == "*":
                    constructor_names.add("Model")

    class CaptureConstructors(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            func = node.func
            constructor = (
                isinstance(func, ast.Name) and func.id in constructor_names
                or isinstance(func, ast.Attribute) and func.attr == "Model"
                and isinstance(func.value, ast.Name) and func.value.id in module_names
            )
            if constructor:
                return ast.copy_location(ast.Call(
                    func=ast.Name(id="__lean_capture_v2", ctx=ast.Load()),
                    args=[node], keywords=[]), node)
            return node

    tree = CaptureConstructors().visit(tree)
    helper = ast.parse(
        "__lean_models_v2 = []\n"
        "def __lean_capture_v2(value):\n"
        "    __lean_models_v2.append(value)\n"
        "    return value\n"
    ).body
    # Keep module docstring and any __future__ imports in their valid positions.
    position = 0
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        position = 1
    while position < len(tree.body) and isinstance(tree.body[position], ast.ImportFrom) and tree.body[position].module == "__future__":
        position += 1
    tree.body[position:position] = helper
    return ast.unparse(ast.fix_missing_locations(tree))


def _unique_models(values):
    result = []
    for value in values:
        if isinstance(value, gp.Model) and all(value is not item for item in result):
            result.append(value)
    return result


def execute_code(code, *, return_trace=False):
    """Execute once, read one model, and preserve interface/solver diagnostics.

    A uniquely captured function-local model may be recovered when the generated
    function omits return. Never create/optimize another model or select by score.
    """
    namespace = {"__name__": "__main__"}
    trace = {"model_interface_version": MODEL_INTERFACE_VERSION,
             "execution_ok": False, "generated_program_completed": False,
             "model_contract_recovered": False, "model_return_strategy": None,
             "model_count": 0, "solver_status": None, "solver_runtime": None,
             "failure_category": None, "executed_source": None}
    models = []
    try:
        source = _source_candidate(code)
        trace["solve_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        executed_source = _instrument_model_creation(source)
        trace["executed_source"] = executed_source
        trace["executed_source_sha256"] = hashlib.sha256(executed_source.encode()).hexdigest()
        exec(compile(executed_source, "<generated-model-interface-v2>", "exec"), namespace)
        trace["generated_program_completed"] = True
        exposed = _unique_models(namespace.values())
        captured = _unique_models(namespace.get("__lean_models_v2", []))
        models = _unique_models([*exposed, *captured])
        trace["model_count"] = len(models)
        if len(models) != 1:
            raise ModelContractError(f"Expected exactly one Gurobi model; found {len(models)}")
        model = models[0]
        if namespace.get("m") is model:
            trace["model_return_strategy"] = "global_m"
        elif namespace.get("model") is model:
            trace["model_return_strategy"] = "global_model"
        elif any(model is item for item in exposed):
            trace.update(model_return_strategy="unique_global_model", model_contract_recovered=True)
        else:
            trace.update(model_return_strategy="captured_constructor", model_contract_recovered=True)
        try:
            trace["solver_status"] = int(model.Status)
            trace["solver_runtime"] = float(model.Runtime)
        except Exception as exc:
            raise ModelContractError("Model is unavailable/disposed before result extraction") from exc
        if trace["solver_status"] != gp.GRB.OPTIMAL:
            raise SolverStatusError(f"No optimal solution; solver status: {trace['solver_status']}")
        objective = float(model.ObjVal)
        if not math.isfinite(objective):
            raise ModelContractError("Solver objective is not finite")
        solution = [(variable.VarName, float(variable.X)) for variable in model.getVars()]
        trace["execution_ok"] = True
        print("Optimal objective:", objective)
        print("Optimal solution:", solution)
        return (objective, solution, trace) if return_trace else (objective, solution)
    except Exception as exc:
        trace["model_count"] = len(_unique_models([*namespace.values(), *namespace.get("__lean_models_v2", [])]))
        trace["failure_category"] = _execution_failure_category(exc)
        exc.execution_trace = trace
        raise
    finally:
        # Release models after reading, including partial programs that raised.
        cleanup = _unique_models([*models, *namespace.values(), *namespace.get("__lean_models_v2", [])])
        for model in cleanup:
            try:
                model.dispose()
            except Exception:
                pass


def _execution_failure_category(exc):
    name = type(exc).__name__
    if isinstance(exc, ModelContractError):
        return "model_interface"
    if isinstance(exc, SolverStatusError):
        return "solver_status"
    if name == "DisabledRouteError":
        return "forbidden_route"
    if name == "ModelOutputTruncated":
        return "output_truncated"
    if name in {"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"}:
        return "model_service"
    if isinstance(exc, SyntaxError):
        return "generated_syntax"
    return "pipeline_error"


def _record_execution_trace(record, trace):
    details = dict(trace)
    source = details.pop("executed_source", None)
    if source is not None:
        record["execution_code"] = source
    record["execution_details"] = json.dumps(details, ensure_ascii=False, allow_nan=False)
    record.update(details)
