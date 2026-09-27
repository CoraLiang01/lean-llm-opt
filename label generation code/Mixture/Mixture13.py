#!/usr/bin/env python3
"""Generate the Mixture13 label model from its benchmark data."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import csv
from decimal import Decimal
from pathlib import Path


DATASET_ADDRESS = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv"
EXPECTED_LABEL_OBJECTIVE = "0.6"
PROBLEM_NAME = "Mixture13"
PROBLEM_CATEGORY = "Mixture"
GUROBI_CL_ENV_VAR = "GUROBI_CL"
STDOUT_END = ''

DAILY_MINUTES = Decimal("1440")


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


def display_number(value: str | Decimal) -> str:
    number = Decimal(str(value))
    if number == number.to_integral_value():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def read_workstations(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError("workstation_times.csv is empty.")

    product_columns = [
        column
        for column in (reader.fieldnames or [])
        if column.startswith("HiFi") and column.endswith("_Minutes")
    ]
    if not product_columns:
        raise ValueError("No HiFi processing-time columns found.")

    required_columns = {"Workstation", "Maintenance_Percent"}
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return rows, product_columns


def term(coefficient: str | Decimal, variable: str) -> str:
    value = Decimal(str(coefficient))
    if value == Decimal(1):
        return variable
    return f"{display_number(value)} {variable}"


def objective_lines(coefficients: list[Decimal], constant: Decimal) -> list[str]:
    lines: list[str] = []
    start = 0
    widths = [7, 7]
    width_index = 0

    while start < len(coefficients):
        width = widths[width_index] if width_index < len(widths) else 6
        chunk = coefficients[start : start + width]
        pieces = [f"- {term(coefficient, f'x[{start + offset}]')}" for offset, coefficient in enumerate(chunk)]
        if start + width >= len(coefficients):
            pieces.append(f"+ {display_number(constant)} Constant")
        lines.append(" ".join(pieces))
        start += width
        width_index += 1

    return lines


def wrap_terms(
    terms: list[str],
    first_prefix: str,
    first_count: int,
    later_count: int,
) -> list[str]:
    lines: list[str] = []
    start = 0
    count = first_count
    while start < len(terms):
        chunk = terms[start : start + count]
        prefix = first_prefix if start == 0 else "+ "
        lines.append(f"{prefix}{' + '.join(chunk)}")
        start += count
        count = later_count
    return lines


def effective_capacity(row: dict[str, str]) -> Decimal:
    maintenance = Decimal(row["Maintenance_Percent"]) / Decimal(100)
    return DAILY_MINUTES * (Decimal(1) - maintenance)


def build_label_model(rows: list[dict[str, str]], product_columns: list[str]) -> str:
    product_count = len(product_columns)
    objective_coefficients = [
        sum(Decimal(row[column]) for row in rows)
        for column in product_columns
    ]
    capacities = [effective_capacity(row) for row in rows]
    constant = sum(capacities)

    lines = [
        r"\ Model MaxBusyTime_MinIdleTime",
        r"    \ LP format - for model browsing. Use MPS format to capture full model detail.",
        "    Minimize",
    ]
    lines.extend(objective_lines(objective_coefficients, constant))
    lines.append("    Subject To")

    for row, capacity in zip(rows, capacities):
        workstation = row["Workstation"]
        constraint_terms = [
            term(row[column], f"x[{index}]")
            for index, column in enumerate(product_columns)
        ]
        constraint_lines = wrap_terms(constraint_terms, f"WS{workstation}: ", 8, 7)
        constraint_lines[-1] = f"{constraint_lines[-1]} <= {display_number(capacity)}"
        lines.extend(constraint_lines)

    lines.extend(["    Bounds", "Constant = 1", "    Generals"])

    variables = [f"x[{index}]" for index in range(product_count)]
    lines.append(" ".join(variables[:14]))
    for start in range(14, product_count, 12):
        lines.append(" ".join(variables[start : start + 12]))

    lines.append("    End")
    return "\n".join(lines) + "\n"


def generate_label_model() -> str:
    workstation_rows, product_time_columns = read_workstations(repo_root() / DATASET_ADDRESS)
    return build_label_model(workstation_rows, product_time_columns)

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
