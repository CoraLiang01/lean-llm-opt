#!/usr/bin/env python3
"""Generate the Mixture14 label model from its benchmark data."""

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
    "Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv\n"
    "Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv"
)
RESOURCE_LIMITS_PATH = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/resource_limits.csv"
PRODUCT_RESOURCES_PATH = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture14/product_resources.csv"
EXPECTED_LABEL_OBJECTIVE = "1344480"
PROBLEM_NAME = "Mixture14"
PROBLEM_CATEGORY = "Mixture"
GUROBI_CL_ENV_VAR = "GUROBI_CL"
STDOUT_END = ''

CATALYST_PRODUCT_INDEX = 3
CATALYST_KG_PER_UNIT = Decimal("5")
CATALYST_SALE_PRICE = Decimal("300")
CATALYST_DISPOSAL_COST = Decimal("200")
CATALYST_SALES_CAP = "1500"


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
    number = Decimal(str(value).strip())
    if number == number.to_integral_value():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def read_resource_limits(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("resource_limits.csv is empty.")

    required_columns = {"Resource", "MonthlyLimit"}
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return {row["Resource"]: display_number(row["MonthlyLimit"]) for row in rows}


def read_products(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("product_resources.csv is empty.")

    required_columns = {"Product", "LaborHours", "MaterialA", "MaterialB", "Profit"}
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return rows


def product_index(product: str) -> int:
    if not product.startswith("Widget"):
        raise ValueError(f"Unexpected product id: {product}")
    return int(product[len("Widget"):])


def product_variable(index: int) -> str:
    return f"x_{index}"


def coefficient_term(coefficient: str | Decimal, variable: str) -> str:
    return f"{display_number(coefficient)}{variable}"


def objective_coefficient(product: dict[str, str]) -> Decimal:
    index = product_index(product["Product"])
    coefficient = Decimal(product["Profit"].strip())
    if index == CATALYST_PRODUCT_INDEX:
        coefficient -= CATALYST_KG_PER_UNIT * CATALYST_DISPOSAL_COST
    return coefficient


def resource_expression(products: list[dict[str, str]], column: str) -> str:
    terms = [
        coefficient_term(product[column], product_variable(product_index(product["Product"])))
        for product in products
    ]
    return " + ".join(terms)


def build_label_model(products: list[dict[str, str]], limits: dict[str, str]) -> str:
    objective_terms = [
        coefficient_term(objective_coefficient(product), product_variable(product_index(product["Product"])))
        for product in products
    ]
    catalyst_sale_coefficient = CATALYST_SALE_PRICE + CATALYST_DISPOSAL_COST
    objective_terms.append(f"{display_number(catalyst_sale_coefficient)}s")

    variable_names = [
        product_variable(product_index(product["Product"]))
        for product in products
    ]

    lines = [
        "#### Decision Variables",
        "    - $x_i$: Units of Widgeti produced",
        "    - $s$: kg of CatalystX sold",
        "",
        "    #### Objective Function",
        "",
        f"    max Z = {' + '.join(objective_terms)}",
        "",
        f"    {resource_expression(products, 'LaborHours')} <= {limits['LaborHours']}",
        "",
        f"    {resource_expression(products, 'MaterialA')} <= {limits['MaterialA']}",
        f"    {resource_expression(products, 'MaterialB')} <= {limits['MaterialB']}",
        f"    s <= {CATALYST_SALES_CAP}",
        f"    s <= {display_number(CATALYST_KG_PER_UNIT)}x_{CATALYST_PRODUCT_INDEX}",
        f"    {', '.join(variable_names)}, s >= 0",
    ]
    return "\n".join(lines) + "\n\n"


def generate_label_model() -> str:
    root = repo_root()
    resource_limits = read_resource_limits(root / RESOURCE_LIMITS_PATH)
    product_rows = read_products(root / PRODUCT_RESOURCES_PATH)
    return build_label_model(product_rows, resource_limits)


def lp_coefficient_term(coefficient: str | Decimal, variable: str) -> str:
    return f"{display_number(coefficient)} {variable}"


def lp_resource_expression(products: list[dict[str, str]], column: str) -> str:
    return " + ".join(
        lp_coefficient_term(product[column], product_variable(product_index(product["Product"])))
        for product in products
    )


def build_lp_model(products: list[dict[str, str]], limits: dict[str, str]) -> str:
    objective_terms = [
        lp_coefficient_term(objective_coefficient(product), product_variable(product_index(product["Product"])))
        for product in products
    ]
    catalyst_sale_coefficient = CATALYST_SALE_PRICE + CATALYST_DISPOSAL_COST
    objective_terms.append(f"{display_number(catalyst_sale_coefficient)} s")

    lines = ["Maximize", f"    obj: {' + '.join(objective_terms)}", "Subject To"]
    lines.append(f"    labor: {lp_resource_expression(products, 'LaborHours')} <= {limits['LaborHours']}")
    lines.append(f"    material_a: {lp_resource_expression(products, 'MaterialA')} <= {limits['MaterialA']}")
    lines.append(f"    material_b: {lp_resource_expression(products, 'MaterialB')} <= {limits['MaterialB']}")
    lines.append(f"    catalyst_sales_cap: s <= {CATALYST_SALES_CAP}")
    lines.append(f"    catalyst_available: s - {display_number(CATALYST_KG_PER_UNIT)} x_{CATALYST_PRODUCT_INDEX} <= 0")
    lines.append("End")
    return "\n".join(lines)


def generate_lp_model() -> str:
    root = repo_root()
    resource_limits = read_resource_limits(root / RESOURCE_LIMITS_PATH)
    product_rows = read_products(root / PRODUCT_RESOURCES_PATH)
    return build_lp_model(product_rows, resource_limits)

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
