#!/usr/bin/env python3
"""Generate the Mixture12 label model from its benchmark data."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import csv
from decimal import Decimal
from pathlib import Path


DATASET_ADDRESS = (
    "Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv\n"
    "Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv\n"
    "Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv"
)
COST_MATRIX_PATH = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv"
DESTINATIONS_PATH = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv"
SOURCES_PATH = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv"
EXPECTED_LABEL_OBJECTIVE = "3148.93"
PROBLEM_NAME = "Mixture12"
PROBLEM_CATEGORY = "Mixture"
GUROBI_CL_ENV_VAR = "GUROBI_CL"
STDOUT_END = ''

TRUCK_CAPACITY = "10"


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


def two_decimal(value: str | Decimal) -> str:
    return f"{Decimal(str(value)):.2f}"


def read_cost_matrix(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError("expanded_cost_matrix.csv is empty.")

    destinations = [column for column in (reader.fieldnames or []) if column != "source_id"]
    return rows, destinations


def read_destinations(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("expanded_destinations.csv is empty.")

    required_columns = {"destination_id", "demand_units"}
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return rows


def read_sources(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("expanded_sources.csv is empty.")

    required_columns = {"source_id", "supply_units"}
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return rows


def numeric_suffix(identifier: str) -> int:
    digits = "".join(character for character in identifier if character.isdigit())
    if not digits:
        raise ValueError(f"Identifier has no numeric suffix: {identifier}")
    return int(digits)


def objective_terms(row: dict[str, str], destinations: list[str]) -> str:
    source_index = numeric_suffix(row["source_id"])
    terms = [
        f"{two_decimal(row[destination])}\\,y_{{{source_index},{numeric_suffix(destination)}}}"
        for destination in destinations
    ]
    return " + ".join(terms)


def demand_variable(destination_index: int) -> str:
    if destination_index <= 9:
        return f"y_{{i{destination_index}}}"
    return f"y_{{i,{destination_index}}}"


def demand_groups(destinations: list[dict[str, str]], source_count: int) -> list[list[str]]:
    groups: list[list[str]] = []
    for start in range(0, len(destinations), 5):
        group: list[str] = []
        for row in destinations[start : start + 5]:
            destination_index = numeric_suffix(row["destination_id"])
            group.append(
                f"\\sum_{{i=1}}^{{{source_count}}} {demand_variable(destination_index)}={row['demand_units']}"
            )
        groups.append(group)
    return groups


def supply_groups(sources: list[dict[str, str]], destination_count: int) -> list[list[str]]:
    groups: list[list[str]] = []
    for start in range(0, len(sources), 5):
        group: list[str] = []
        for row in sources[start : start + 5]:
            source_index = numeric_suffix(row["source_id"])
            group.append(
                f"\\sum_{{j=1}}^{{{destination_count}}} y_{{{source_index}j}}\\le {row['supply_units']}"
            )
        groups.append(group)
    return groups


def append_constraint_groups(lines: list[str], groups: list[list[str]], final_period: bool = False) -> None:
    for group_index, group in enumerate(groups):
        lines.append("$$")
        for item_index, item in enumerate(group):
            is_last_item = group_index == len(groups) - 1 and item_index == len(group) - 1
            punctuation = "." if final_period and is_last_item else ","
            lines.append(f"{item}{punctuation}\\quad")
        if lines[-1].endswith("\\quad"):
            lines[-1] = lines[-1][:-len("\\quad")]
        lines.append("$$")


def build_label_model(
    cost_rows: list[dict[str, str]],
    destinations: list[str],
    destination_rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
) -> str:
    source_count = len(cost_rows)
    destination_count = len(destinations)

    lines = [
        "**Objective Function:**",
        "",
        "$$",
        "\\min\\ Z\\;=\\;",
    ]

    for index, row in enumerate(cost_rows):
        lines.extend(
            [
                "\\bigl(",
                objective_terms(row, destinations),
                "\\bigr)",
            ]
        )
        if index < len(cost_rows) - 1:
            lines.append("\\;+\\;")

    lines.extend(
        [
            "",
            "$$",
            "",
            "**Constraints:**",
            "",
            "% Demand satisfaction (fixed demands, units of cargo)",
        ]
    )

    append_constraint_groups(lines, demand_groups(destination_rows, source_count), final_period=True)

    lines.extend(
        [
            "",
            "% Supply limits (each source cannot exceed its available supply)",
        ]
    )
    append_constraint_groups(lines, supply_groups(source_rows, destination_count), final_period=True)

    lines.extend(
        [
            "",
            "% Truck-capacity coupling (integer trucks, partial loading allowed)",
            "$$",
            f"y_{{ij}}\\ \\le\\ {TRUCK_CAPACITY}\\,x_{{ij}},\\qquad \\forall\\, i=1,\\dots,{source_count},\\ \\forall\\, j=1,\\dots,{destination_count}.",
            "$$",
            "",
            "**Decision Variables:**",
            "$$",
            "y_{ij}\\ \\ge 0 \\quad\\text{(units shipped from S}i\\text{ to D}j\\text{)},\\qquad",
            "x_{ij}\\in \\mathbb{Z}_{+}\\quad\\text{(number of trucks on route }i\\!\\to\\! j\\text{)}.",
            "$$",
        ]
    )
    return "\n".join(lines)


def generate_label_model() -> str:
    root = repo_root()
    costs, destination_ids = read_cost_matrix(root / COST_MATRIX_PATH)
    destination_data = read_destinations(root / DESTINATIONS_PATH)
    source_data = read_sources(root / SOURCES_PATH)
    return build_label_model(costs, destination_ids, destination_data, source_data)


def build_lp_model(
    cost_rows: list[dict[str, str]],
    destinations: list[str],
    destination_rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
) -> str:
    objective = []
    for row in cost_rows:
        source_index = numeric_suffix(row["source_id"])
        for destination in destinations:
            objective.append(f"{two_decimal(row[destination])} y_{source_index}_{numeric_suffix(destination)}")

    lines = ["Minimize", f"    obj: {' + '.join(objective)}", "Subject To"]
    source_count = len(source_rows)
    destination_count = len(destination_rows)
    for row in destination_rows:
        j = numeric_suffix(row["destination_id"])
        lines.append(
            f"    demand_{j}: "
            + " + ".join(f"y_{i}_{j}" for i in range(1, source_count + 1))
            + f" = {row['demand_units']}"
        )
    for row in source_rows:
        i = numeric_suffix(row["source_id"])
        lines.append(
            f"    supply_{i}: "
            + " + ".join(f"y_{i}_{j}" for j in range(1, destination_count + 1))
            + f" <= {row['supply_units']}"
        )
    for i in range(1, source_count + 1):
        for j in range(1, destination_count + 1):
            lines.append(f"    truck_link_{i}_{j}: y_{i}_{j} - {TRUCK_CAPACITY} x_{i}_{j} <= 0")
    lines.extend(["Generals", "    " + " ".join(f"x_{i}_{j}" for i in range(1, source_count + 1) for j in range(1, destination_count + 1)), "End"])
    return "\n".join(lines)


def generate_lp_model() -> str:
    root = repo_root()
    costs, destination_ids = read_cost_matrix(root / COST_MATRIX_PATH)
    destination_data = read_destinations(root / DESTINATIONS_PATH)
    source_data = read_sources(root / SOURCES_PATH)
    return build_lp_model(costs, destination_ids, destination_data, source_data)

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
