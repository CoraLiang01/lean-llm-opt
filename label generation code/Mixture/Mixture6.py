#!/usr/bin/env python3
"""Generate the Mixture6 label model from its benchmark data."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import csv
from decimal import Decimal
from pathlib import Path


DATASET_ADDRESS = "Test_Dataset/Large-scale-or//Mixture_testing/Mixture6/parameters.csv"
EXPECTED_LABEL_OBJECTIVE = "14300"
PROBLEM_NAME = "Mixture6"
PROBLEM_CATEGORY = "Mixture"
GUROBI_CL_ENV_VAR = "GUROBI_CL"
STDOUT_END = '\n'

MIN_RUNTIME_PERIODS = 2
LOAD_FLUCTUATION_LIMIT = 300
SPARE_CAPACITY_FACTOR = "0.9"


def script_path() -> Path | None:
    try:
        return Path(__file__).resolve()
    except NameError:
        return None


def repo_root() -> Path:
    script = script_path()
    candidates = []
    if script is not None:
        candidates.append(script.parents[2])
    candidates.extend([Path.cwd(), *Path.cwd().parents])

    for candidate in candidates:
        if (candidate / DATASET_ADDRESS.splitlines()[0]).exists():
            return candidate
    return candidates[0]


def default_output_dir() -> Path:
    script = script_path()
    if script is not None:
        return script.parent
    return repo_root() / "label generation code" / PROBLEM_CATEGORY


def dataset_path(root: Path) -> Path:
    return root / DATASET_ADDRESS.replace("//", "/")


def read_parameters(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("parameters.csv is empty.")

    required_columns = {"truck_id", "Q", "S", "C", "d1", "d2", "d3", "d4"}
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    demand_columns = [column for column in rows[0] if column.startswith("d")]
    demands = [rows[0][column] for column in demand_columns]
    for row in rows[1:]:
        if [row[column] for column in demand_columns] != demands:
            raise ValueError("Demand columns must be identical for every truck row.")

    return rows, demands


def build_label_model(trucks: list[dict[str, str]], demands: list[str]) -> str:
    truck_count = len(trucks)
    period_count = len(demands)
    startup_periods = ", ".join(str(period) for period in range(2, period_count + 1))
    runtime_start_periods = ", ".join(str(period) for period in range(2, period_count))

    lines = [
        "### Sets and Parameters",
        "",
        f"    * **$I = \\{{1, \\dots, {truck_count}\\}}$**: Set of available trucks.",
        f"    * **$T = \\{{1, \\dots, {period_count}\\}}$**: Set of time periods.",
        "    * **$d_t$**: Customer demand in period $t$ ($\\text{kg}$).",
        "    * **$Q_i$**: Maximum capacity of truck $i$ ($\\text{kg}$).",
        "    * **$S_i$**: Startup cost for truck $i$.",
        "    * **$C_i$**: Unit transportation cost for truck $i$.",
        "",
        "    ### Decision Variables",
        "",
        "    * **$w_{it} \\in \\mathbb{R}^+$**: Weight transported by truck $i$ in period $t$. (Continuous)",
        "    * **$y_{it} \\in \\{0, 1\\}$**: Operating status ($1$ if running, $0$ otherwise). (Binary)",
        f"    * **$u_{{it}} \\in \\{{0, 1\\}}$**: Startup status in period $t$ (defined for $t \\in \\{{{startup_periods}\\}}$). (Binary)",
        "",
        "    ### Objective Function (Minimize Total Cost)",
        "",
        f"    $$\\min \\sum_{{i \\in I}} \\left( \\sum_{{t=1}}^{{{period_count}}} C_i w_{{it}} \\right) + \\sum_{{i \\in I}} \\left( S_i y_{{i1}} + \\sum_{{t=2}}^{{{period_count}}} S_i u_{{it}} \\right)$$",
        "",
        "    ---",
        "",
        "    ### Constraints",
        "",
        "    #### 1. Demand Satisfaction",
        "    The total weight transported must meet the demand $d_t$ in every period.",
        "    $$\\sum_{i \\in I} w_{it} \\geq d_t, \\quad \\forall t \\in T$$",
        "",
        "    #### 2. Capacity and Status Linkage",
        "    The transported weight cannot exceed the capacity $Q_i$ when the truck is running, and must be zero if the truck is off.",
        "    $$w_{it} \\leq Q_i y_{it}, \\quad \\forall i \\in I, t \\in T$$",
        "",
        "    #### 3. Spare Capacity Buffer (10% Reserve)",
        "    Total load cannot exceed $90\\%$ of the active trucks' total capacity.",
        f"    $$\\sum_{{i \\in I}} w_{{it}} \\leq {SPARE_CAPACITY_FACTOR} \\sum_{{i \\in I}} Q_i y_{{it}}, \\quad \\forall t \\in T$$",
        "",
        "    #### 4. Startup Variable Definition",
        "    These constraints ensure $u_{it}=1$ only when a truck switches from OFF ($y_{i,t-1}=0$) to ON ($y_{it}=1$).",
        "",
        f"    * **Lower Bound**: $u_{{it}} \\geq y_{{it}} - y_{{i,t-1}}, \\quad \\forall i \\in I, t \\in \\{{{startup_periods}\\}}$",
        f"    * **Upper Bound 1**: $u_{{it}} \\leq y_{{it}}, \\quad \\forall i \\in I, t \\in \\{{{startup_periods}\\}}$",
        f"    * **Upper Bound 2**: $u_{{it}} \\leq 1 - y_{{i,t-1}}, \\quad \\forall i \\in I, t \\in \\{{{startup_periods}\\}}$",
        "",
        f"    #### 5. Minimum Runtime ({MIN_RUNTIME_PERIODS} Periods)",
        f"    Once a truck starts, it must run for at least {MIN_RUNTIME_PERIODS} consecutive periods.",
        "",
        "    * **Start at $t=1$**: $y_{i1} \\leq y_{i2}, \\quad \\forall i \\in I$",
        f"    * **Start at $t \\geq 2$**: $y_{{it}} + y_{{i,t+1}} \\geq 2 u_{{it}}, \\quad \\forall i \\in I, t \\in \\{{{runtime_start_periods}\\}}$",
        f"    * **No start at $t={period_count}$**: $u_{{i{period_count}}} = 0, \\quad \\forall i \\in I$",
        "",
        "    #### 6. Cooldown Constraints (1 Period Off)",
        "    If a truck is shut down, it must remain idle for the following period.",
        f"    $$y_{{i,t-1}} - y_{{it}} \\leq 1 - y_{{i,t+1}}, \\quad \\forall i \\in I, t \\in \\{{{runtime_start_periods}\\}}$$",
        "",
        "    #### 7. Load Fluctuation ($\\pm 300 \\text{ kg}$)",
        f"    The change in load between adjacent periods cannot exceed ${LOAD_FLUCTUATION_LIMIT} \\text{{ kg}}$.",
        "",
        f"    * **Ramp Up**: $w_{{it}} - w_{{i,t-1}} \\leq {LOAD_FLUCTUATION_LIMIT}, \\quad \\forall i \\in I, t \\in \\{{{startup_periods}\\}}$",
        f"    * **Ramp Down**: $w_{{i,t-1}} - w_{{it}} \\leq {LOAD_FLUCTUATION_LIMIT}, \\quad \\forall i \\in I, t \\in \\{{{startup_periods}\\}}$",
        "",
        "    #### 8. Variable Domains",
        "    * $w_{it} \\geq 0, \\quad \\forall i \\in I, t \\in T$",
        "    * $y_{it} \\in \\{0, 1\\}, \\quad \\forall i \\in I, t \\in T$",
        f"    * $u_{{it}} \\in \\{{0, 1\\}}, \\quad \\forall i \\in I, t \\in \\{{{startup_periods}\\}}$",
    ]
    return "\n".join(lines)


def generate_label_model() -> str:
    truck_rows, period_demands = read_parameters(dataset_path(repo_root()))
    return build_label_model(truck_rows, period_demands)


def lp_expression(terms: list[tuple[str, str]]) -> str:
    pieces: list[str] = []
    for coefficient, variable in terms:
        coef = Decimal(str(coefficient))
        if coef == 0:
            continue
        magnitude = display_number(abs(coef)) if "display_number" in globals() else str(abs(coef))
        body = variable if magnitude == "1" else f"{magnitude} {variable}"
        if not pieces:
            pieces.append(f"- {body}" if coef < 0 else body)
        else:
            pieces.append(f"- {body}" if coef < 0 else f"+ {body}")
    return " ".join(pieces) if pieces else "0"


def build_lp_model(trucks: list[dict[str, str]], demands: list[str]) -> str:
    period_count = len(demands)
    objective: list[tuple[str, str]] = []
    for i, truck in enumerate(trucks, start=1):
        for t in range(1, period_count + 1):
            objective.append((truck["C"], f"w_{i}_{t}"))
        objective.append((truck["S"], f"y_{i}_1"))
        for t in range(2, period_count + 1):
            objective.append((truck["S"], f"u_{i}_{t}"))

    lines = ["Minimize", f"    obj: {lp_expression(objective)}", "Subject To"]
    for t, demand in enumerate(demands, start=1):
        lines.append(
            f"    demand_{t}: "
            + lp_expression([("1", f"w_{i}_{t}") for i in range(1, len(trucks) + 1)])
            + f" >= {demand}"
        )
    for i, truck in enumerate(trucks, start=1):
        for t in range(1, period_count + 1):
            lines.append(f"    capacity_link_{i}_{t}: w_{i}_{t} - {truck['Q']} y_{i}_{t} <= 0")
    for t in range(1, period_count + 1):
        terms = [("1", f"w_{i}_{t}") for i in range(1, len(trucks) + 1)]
        terms += [(str(-Decimal(SPARE_CAPACITY_FACTOR) * Decimal(truck["Q"])), f"y_{i}_{t}") for i, truck in enumerate(trucks, start=1)]
        lines.append(f"    spare_capacity_{t}: {lp_expression(terms)} <= 0")
    for i in range(1, len(trucks) + 1):
        for t in range(2, period_count + 1):
            lines.append(f"    startup_lb_{i}_{t}: y_{i}_{t} - y_{i}_{t - 1} - u_{i}_{t} <= 0")
            lines.append(f"    startup_ub_status_{i}_{t}: u_{i}_{t} - y_{i}_{t} <= 0")
            lines.append(f"    startup_ub_prev_{i}_{t}: u_{i}_{t} + y_{i}_{t - 1} <= 1")
        lines.append(f"    min_runtime_start_1_{i}: y_{i}_1 - y_{i}_2 <= 0")
        for t in range(2, period_count):
            lines.append(f"    min_runtime_{i}_{t}: 2 u_{i}_{t} - y_{i}_{t} - y_{i}_{t + 1} <= 0")
        lines.append(f"    no_start_last_{i}: u_{i}_{period_count} = 0")
        for t in range(2, period_count):
            lines.append(f"    cooldown_{i}_{t}: y_{i}_{t - 1} - y_{i}_{t} + y_{i}_{t + 1} <= 1")
        for t in range(2, period_count + 1):
            lines.append(f"    ramp_up_{i}_{t}: w_{i}_{t} - w_{i}_{t - 1} <= {LOAD_FLUCTUATION_LIMIT}")
            lines.append(f"    ramp_down_{i}_{t}: w_{i}_{t - 1} - w_{i}_{t} <= {LOAD_FLUCTUATION_LIMIT}")

    binaries = [f"y_{i}_{t}" for i in range(1, len(trucks) + 1) for t in range(1, period_count + 1)]
    binaries += [f"u_{i}_{t}" for i in range(1, len(trucks) + 1) for t in range(2, period_count + 1)]
    lines.extend(["Binaries", "    " + " ".join(binaries), "End"])
    return "\n".join(lines)


def generate_lp_model() -> str:
    truck_rows, period_demands = read_parameters(dataset_path(repo_root()))
    return build_lp_model(truck_rows, period_demands)

def format_objective(value: float) -> str:
    return f"{value:.12g}"


def expected_objective_components() -> tuple[float, int] | None:
    cleaned = EXPECTED_LABEL_OBJECTIVE.replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", cleaned)
    if match is None:
        return None
    token = match.group(0)
    mantissa = re.split(r"[eE]", token, maxsplit=1)[0]
    decimals = len(mantissa.split(".", maxsplit=1)[1]) if "." in mantissa else 0
    return float(token), decimals


def objective_matches_expected(objective: str | None) -> bool:
    if objective is None:
        return False
    expected = expected_objective_components()
    if expected is None:
        return False
    expected_value, decimals = expected
    actual = float(objective)
    decimal_tolerance = 0.5 * 10 ** (-decimals) if decimals > 0 else 1e-6
    tolerance = max(1e-6, abs(expected_value) * 1e-9, decimal_tolerance)
    return abs(actual - expected_value) <= tolerance


def parse_gurobi_cli_result(output: str) -> tuple[str, str | None]:
    optimal_match = re.search(r"^Optimal objective\s+([-+0-9.eE]+)", output, re.MULTILINE)
    if optimal_match:
        return "OPTIMAL", format_objective(float(optimal_match.group(1)))

    best_match = re.search(r"^Best objective\s+([-+0-9.eE]+)", output, re.MULTILINE)
    if best_match and re.search(r"^Optimal solution found", output, re.MULTILINE):
        return "OPTIMAL", format_objective(float(best_match.group(1)))

    status_patterns = [
        ("INFEASIBLE", r"^Infeasible model"),
        ("UNBOUNDED", r"^Unbounded model"),
        ("INF_OR_UNBD", r"^Model is infeasible or unbounded"),
        ("TIME_LIMIT", r"^Time limit reached"),
    ]
    for status, pattern in status_patterns:
        if re.search(pattern, output, re.MULTILINE):
            return status, None
    return "UNKNOWN", None


def solve_and_write_mps(lp_path: Path, mps_path: Path) -> tuple[str, str | None]:
    try:
        import gurobipy as gp  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        gurobi_cl = os.environ.get(GUROBI_CL_ENV_VAR) or shutil.which("gurobi_cl")
        if gurobi_cl is None:
            raise RuntimeError(
                "Could not solve/write MPS because neither gurobipy nor gurobi_cl is available. "
                "Install gurobipy in this Python environment, put gurobi_cl on PATH, or set "
                f"{GUROBI_CL_ENV_VAR} to the gurobi_cl executable path."
            )
        env = os.environ.copy()
        env.setdefault("LC_ALL", "C")
        env.setdefault("LANG", "C")
        result = subprocess.run(
            [gurobi_cl, "LogFile=", f"ResultFile={mps_path}", str(lp_path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "gurobi_cl failed while solving/writing the MPS file:\n"
                f"{result.stdout}"
            )
        return parse_gurobi_cli_result(result.stdout)

    model = gp.read(str(lp_path))
    model.Params.OutputFlag = 0
    model.optimize()
    model.write(str(mps_path))
    status_name = {
        gp.GRB.OPTIMAL: "OPTIMAL",
        gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.UNBOUNDED: "UNBOUNDED",
        gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
    }.get(model.Status, str(model.Status))
    objective = format_objective(model.ObjVal) if model.SolCount else None
    return status_name, objective


def lp_model_for_files(label_model: str) -> str:
    if "generate_lp_model" in globals():
        return generate_lp_model()
    return label_model


def save_model_files(
    label_model: str,
    output_dir: Path,
    write_mps: bool = True,
) -> tuple[Path, Path | None, tuple[str, str | None] | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lp_path = output_dir / f"{PROBLEM_NAME}.lp"
    mps_path = output_dir / f"{PROBLEM_NAME}.mps"
    lp_path.write_text(lp_model_for_files(label_model) + "\n", encoding="utf-8")
    if write_mps:
        solve_result = solve_and_write_mps(lp_path, mps_path)
        return lp_path, mps_path, solve_result
    return lp_path, None, None


def print_generation_summary(
    lp_path: Path,
    mps_path: Path | None,
    solve_result: tuple[str, str | None] | None,
) -> None:
    print()
    print("Gurobi solve result")
    print(f"LP file: {lp_path}")
    if mps_path is None or solve_result is None:
        print("MPS file: skipped")
        print("Status: skipped")
        return

    status, objective = solve_result
    print(f"MPS file: {mps_path}")
    print(f"Status: {status}")
    if objective is not None:
        print(f"Objective: {objective}")
        print(f"Expected objective: {EXPECTED_LABEL_OBJECTIVE}")
        print(f"Matches expected: {objective_matches_expected(objective)}")


def generate_files(
    output_dir: Path | None = None,
    write_mps: bool = True,
) -> tuple[str, Path, Path | None, tuple[str, str | None] | None]:
    label_model = generate_label_model()
    lp_path, mps_path, solve_result = save_model_files(
        label_model,
        output_dir or default_output_dir(),
        write_mps=write_mps,
    )
    return label_model, lp_path, mps_path, solve_result


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Generate the {PROBLEM_NAME} label model.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help=f"Directory where {PROBLEM_NAME}.lp and {PROBLEM_NAME}.mps will be written.",
    )
    parser.add_argument(
        "--no-files",
        action="store_true",
        help="Only print the label model; do not write LP or MPS files.",
    )
    parser.add_argument(
        "--skip-mps",
        action="store_true",
        help=f"Write {PROBLEM_NAME}.lp only; skip {PROBLEM_NAME}.mps generation.",
    )
    args, _ = parser.parse_known_args()

    label_model = generate_label_model()
    file_result = None
    if not args.no_files:
        file_result = save_model_files(label_model, args.output_dir, write_mps=not args.skip_mps)
    print(label_model, end=STDOUT_END)
    if file_result is not None:
        print_generation_summary(*file_result)


if __name__ == "__main__":
    main()
