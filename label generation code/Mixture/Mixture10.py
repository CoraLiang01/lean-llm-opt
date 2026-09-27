#!/usr/bin/env python3
"""Generate the Mixture10 label model from its benchmark data."""

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
    "Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv\n"
    "Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv"
)
PRODUCTS_PATH = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/factory_products_100.csv"
RESOURCES_PATH = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture10/resources_capacities.csv"
EXPECTED_LABEL_OBJECTIVE = "108657.9"
PROBLEM_NAME = "Mixture10"
PROBLEM_CATEGORY = "Mixture"
GUROBI_CL_ENV_VAR = "GUROBI_CL"
STDOUT_END = ''



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


def read_products(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("factory_products_100.csv is empty.")

    required_columns = {
        "product",
        "profit_per_unit",
        "r1_per_unit",
        "r2_per_unit",
        "r3_per_unit",
        "upper_demand_units",
        "batch_size_units",
    }
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return rows


def read_resources(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("resources_capacities.csv is empty.")

    required_columns = {"resource", "capacity"}
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return {row["resource"]: row["capacity"] for row in rows}


def product_index(product: str) -> int:
    if not product.startswith("P"):
        raise ValueError(f"Unexpected product id: {product}")
    return int(product[1:])


def two_decimal(value: str | Decimal) -> str:
    return f"{Decimal(str(value)):.2f}"


def scaled(value: str, batch_size: str) -> Decimal:
    return Decimal(value) * Decimal(batch_size)


def terms_for(
    products: list[dict[str, str]],
    coefficient_column: str,
) -> str:
    terms: list[str] = []
    for row in products:
        index = product_index(row["product"])
        coefficient = scaled(row[coefficient_column], row["batch_size_units"])
        terms.append(f"{two_decimal(coefficient)} x_{index}")
    return " + ".join(terms)


def build_label_model(products: list[dict[str, str]], resources: dict[str, str]) -> str:
    product_count = len(products)

    lines = [
        "Mathematical Formulation for the 100-Product Batch Production Problem",
        "",
        "Sets",
        f"I = {{1,2,...,{product_count}}}   (products P1-P{product_count})",
        "",
        "Decision Variables",
        "For each i in I, let x_i = number of batches of product P_i produced, where each batch = 10 units.",
        "x_i is a nonnegative integer for all i in I.",
        "",
        "Objective Function",
        "Maximize",
        f"Z = {terms_for(products, 'profit_per_unit')}",
        "",
        "Subject to",
        "",
    ]

    for resource, column in (("R1", "r1_per_unit"), ("R2", "r2_per_unit"), ("R3", "r3_per_unit")):
        lines.append(f"Resource {resource}:")
        lines.append(f"{terms_for(products, column)} <= {resources[resource]}")
        lines.append("")

    lines.append("Demand upper bound constraints:")
    for row in products:
        index = product_index(row["product"])
        batch_size = row["batch_size_units"]
        demand = row["upper_demand_units"]
        lines.append(f"{batch_size} x_{index} <= {demand}")

    lines.extend(
        [
            "",
            "Integrality and nonnegativity:",
            f"x_i in Z_+, for all i = 1,...,{product_count}",
        ]
    )
    return "\n".join(lines)


def generate_label_model() -> str:
    root = repo_root()
    product_rows = read_products(root / PRODUCTS_PATH)
    resource_capacities = read_resources(root / RESOURCES_PATH)
    return build_label_model(product_rows, resource_capacities)


def build_lp_model(products: list[dict[str, str]], resources: dict[str, str]) -> str:
    lines = ["Maximize", f"    obj: {terms_for(products, 'profit_per_unit')}", "Subject To"]
    for resource, column in (("R1", "r1_per_unit"), ("R2", "r2_per_unit"), ("R3", "r3_per_unit")):
        lines.append(f"    resource_{resource}: {terms_for(products, column)} <= {resources[resource]}")
    for row in products:
        index = product_index(row["product"])
        lines.append(f"    demand_{index}: {row['batch_size_units']} x_{index} <= {row['upper_demand_units']}")
    lines.extend(["Generals", "    " + " ".join(f"x_{product_index(row['product'])}" for row in products), "End"])
    return "\n".join(lines)


def generate_lp_model() -> str:
    root = repo_root()
    product_rows = read_products(root / PRODUCTS_PATH)
    resource_capacities = read_resources(root / RESOURCES_PATH)
    return build_lp_model(product_rows, resource_capacities)

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
