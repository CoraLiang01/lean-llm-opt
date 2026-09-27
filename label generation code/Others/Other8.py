#!/usr/bin/env python3
"""Generate the Other8 label model from its benchmark data."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import csv
from pathlib import Path


DATASET_ADDRESS = (
    "Test_Dataset/Large-scale-or/Others_example/Others8/processing_time_unit.csv\n"
    "Test_Dataset/Large-scale-or/Others_example/Others8/unit_price.csv\n"
    "Test_Dataset/Large-scale-or/Others_example/Others8/total_working_hours.csv"
)
PROCESSING_TIME_PATH = (
    "Test_Dataset/Large-scale-or/Others_example/Others8/processing_time_unit.csv"
)
UNIT_PRICE_PATH = "Test_Dataset/Large-scale-or/Others_example/Others8/unit_price.csv"
TOTAL_WORKING_HOURS_PATH = (
    "Test_Dataset/Large-scale-or/Others_example/Others8/total_working_hours.csv"
)
EXPECTED_LABEL_OBJECTIVE = "1382200.722"
PROBLEM_NAME = "Other8"
PROBLEM_CATEGORY = "Others"
GUROBI_CL_ENV_VAR = "GUROBI_CL"
STDOUT_END = '\n'



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


def read_unit_prices(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows or "unit_price" not in rows[0]:
        raise ValueError("unit_price.csv must contain a 'unit_price' column.")
    return [row["unit_price"] for row in rows]


def read_total_hours(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows or "total_hours" not in rows[0]:
        raise ValueError(
            "total_working_hours.csv must contain a 'total_hours' column."
        )
    return [row["total_hours"] for row in rows]


def read_processing_times(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 2:
        raise ValueError("processing_time_unit.csv is empty.")
    return [row[1:] for row in rows[1:]]


def bracket_list(values: list[str]) -> str:
    return f"[{', '.join(values)}]"


def build_label_model(
    unit_prices: list[str],
    total_hours: list[str],
    processing_times: list[list[str]],
) -> str:
    component_count = len(unit_prices)
    workshop_count = len(total_hours)

    if len(processing_times) != workshop_count:
        raise ValueError("Processing-time rows and total-hour rows must match.")
    for row_number, row in enumerate(processing_times, start=1):
        if len(row) != component_count:
            raise ValueError(
                f"Processing-time row {row_number} has {len(row)} values, "
                f"expected {component_count}."
            )

    lines = [
        "**Indices:**",
        f"    * $j$: Index for components, from 1 to {component_count}.",
        f"    * $i$: Index for workshops, from 1 to {workshop_count}.",
        "",
        "    **Parameters:**",
        "    * $p_j$: The unit price of component $j$.",
        "    * $t_{ij}$: The time (in hours) required in workshop $i$ to process one unit of component $j$.",
        "    * $H_i$: The total hours of processing time available in workshop $i$.",
        "",
        "    **Decision Variable:**",
        "    * $x_j \\geq 0$: The number of units of component $j$ to produce. This is a continuous variable.",
        "    ** Data***",
        "    $$",
        f"    unit_prices = {bracket_list(unit_prices)}",
        "    $$",
        "",
        "    $$",
        f"    total_hours = {bracket_list(total_hours)}",
        "    $$",
        "",
        "    $$",
        "    processing_times = [",
        "",
    ]

    for index, row in enumerate(processing_times):
        suffix = "," if index < len(processing_times) - 1 else "                "
        lines.append(f"    {bracket_list(row)}{suffix}")

    lines.extend(
        [
            "    ]",
            "    $$",
            "    ---",
            "",
            "    ### **Objective Function (Maximize Total Revenue):**",
            "    The goal is to maximize the total revenue, calculated as the sum of the quantities of each component produced multiplied by their respective unit prices.",
            "    $$",
            f"    \\text{{Maximize}} \\quad Z = \\sum_{{j=1}}^{{{component_count}}} p_j x_j",
            "    $$",
            "",
            "    ---",
            "",
            "    ### **Subject to:**",
            "",
            "    **1. Workshop Time Constraints:**",
            "    For each workshop $i$, the total time consumed by producing all components cannot exceed the total available hours in that workshop.",
            "    $$",
            f"    \\sum_{{j=1}}^{{{component_count}}} t_{{ij}} x_j \\leq H_i \\quad \\forall i \\in \\{{1, \\ldots, {workshop_count}\\}}",
            "    $$",
            "",
            "    **2. Non-Negativity Constraint:**",
            "    The quantity produced of any component cannot be negative.",
            "    $$",
            f"    x_j \\geq 0 \\quad \\forall j \\in \\{{1, \\ldots, {component_count}\\}}",
            "    $$",
        ]
    )
    return "\n".join(lines)


def generate_label_model() -> str:
    root = repo_root()
    unit_price_list = read_unit_prices(root / UNIT_PRICE_PATH)
    total_hour_list = read_total_hours(root / TOTAL_WORKING_HOURS_PATH)
    processing_time_matrix = read_processing_times(root / PROCESSING_TIME_PATH)
    return build_label_model(unit_price_list, total_hour_list, processing_time_matrix)


def lp_expression(terms: list[tuple[str, str]]) -> str:
    return " + ".join(f"{coef} {var}" if coef != "1" else var for coef, var in terms)


def build_lp_model(
    unit_prices: list[str],
    total_hours: list[str],
    processing_times: list[list[str]],
) -> str:
    component_count = len(unit_prices)
    lines = ["Maximize"]
    lines.append(
        "    obj: "
        + lp_expression([(unit_prices[j], f"x_{j + 1}") for j in range(component_count)])
    )
    lines.append("Subject To")
    for i, row in enumerate(processing_times, start=1):
        terms = [(coef, f"x_{j + 1}") for j, coef in enumerate(row)]
        lines.append(f"    workshop_time_{i}: {lp_expression(terms)} <= {total_hours[i - 1]}")
    lines.append("End")
    return "\n".join(lines)


def generate_lp_model() -> str:
    root = repo_root()
    unit_price_list = read_unit_prices(root / UNIT_PRICE_PATH)
    total_hour_list = read_total_hours(root / TOTAL_WORKING_HOURS_PATH)
    processing_time_matrix = read_processing_times(root / PROCESSING_TIME_PATH)
    return build_lp_model(unit_price_list, total_hour_list, processing_time_matrix)

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
