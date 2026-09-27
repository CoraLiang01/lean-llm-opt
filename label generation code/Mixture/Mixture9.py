#!/usr/bin/env python3
"""Generate the Mixture9 label model from its benchmark data."""

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
    "Test_Dataset/Large-scale-or//Mixture_testing/Mixture9/36-1.csv\n"
    "Test_Dataset/Large-scale-or//Mixture_testing/Mixture9/36-2.csv\n"
    "Test_Dataset/Large-scale-or//Mixture_testing/Mixture9/36-3.csv"
)
PRODUCT_DATA_PATH = "Test_Dataset/Large-scale-or//Mixture_testing/Mixture9/36-1.csv"
ACTIVATION_COST_PATH = "Test_Dataset/Large-scale-or//Mixture_testing/Mixture9/36-2.csv"
MIN_BATCH_PATH = "Test_Dataset/Large-scale-or//Mixture_testing/Mixture9/36-3.csv"
EXPECTED_LABEL_OBJECTIVE = "6,243,055.96"
PROBLEM_NAME = "Mixture9"
PROBLEM_CATEGORY = "Mixture"
GUROBI_CL_ENV_VAR = "GUROBI_CL"
STDOUT_END = '\n'

PRODUCTION_DAYS = 22


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


def clean_path(path: str) -> str:
    return path.replace("//", "/")


def display_number(value: str | Decimal) -> str:
    number = Decimal(str(value))
    if number == number.to_integral_value():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def latex_int(value: str | int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def read_transposed_table(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"{path.name} is empty.")

    products = [column for column in rows[0] if column != "Product"]
    data = {row["Product"]: {product: row[product] for product in products} for row in rows}
    return products, data


def product_index(product: str) -> int:
    if not product.startswith("A"):
        raise ValueError(f"Unexpected product id: {product}")
    return int(product[1:])


def append_scalar_constraint(
    lines: list[str],
    formula: str,
    description: str,
) -> None:
    lines.extend(
        [
            "    $$",
            f"    {formula}",
            f"    \\quad (\\text{{{description}}})",
            "    $$",
            "",
        ]
    )


def build_label_model(
    products: list[str],
    product_data: dict[str, dict[str, str]],
    activation_costs: dict[str, str],
    min_batches: dict[str, str],
) -> str:
    demand = product_data["Maximum Demand (100 kg units)"]
    prices = product_data["Selling Price ($/100 kg)"]
    production_costs = product_data["Production Cost ($/100 kg)"]
    daily_quotas = product_data["Production Quota (max per day)"]

    lines = [
        "## Mathematical Model",
        "",
        "    **Maximize**",
        "",
        "    $$",
    ]

    for offset, product in enumerate(products):
        index = product_index(product)
        unit_profit = Decimal(prices[product]) - Decimal(production_costs[product])
        fixed_cost = latex_int(activation_costs[product])
        prefix = "    " if offset == 0 else "    + "
        trailing_space = " " if offset < len(products) - 1 else ""
        lines.append(
            f"{prefix}{display_number(unit_profit)}\\, x_{index} - {fixed_cost}\\, y_{index}{trailing_space}"
        )

    lines.extend(
        [
            "    $$",
            "",
            "    **subject to**",
            "",
        ]
    )

    for product in products:
        index = product_index(product)
        append_scalar_constraint(
            lines,
            f"x_{index} \\leq {latex_int(demand[product])}",
            f"product {index} demand",
        )

    for product in products:
        index = product_index(product)
        monthly_quota = int(daily_quotas[product]) * PRODUCTION_DAYS
        append_scalar_constraint(
            lines,
            f"x_{index} \\leq {latex_int(monthly_quota)}",
            f"product {index} monthly quota",
        )

    for product in products:
        index = product_index(product)
        append_scalar_constraint(
            lines,
            f"x_{index} \\geq {latex_int(min_batches[product])}\\, y_{index}",
            f"minimum batch for product {index}",
        )

    for product in products:
        index = product_index(product)
        append_scalar_constraint(
            lines,
            f"x_{index} \\leq {latex_int(demand[product])}\\, y_{index}",
            f"linking product {index} activation",
        )

    x_variables = ", ".join(f"x_{product_index(product)}" for product in products)
    y_variables = ", ".join(f"y_{product_index(product)}" for product in products)

    lines.extend(
        [
            "    **where**",
            "",
            "    $$",
            f"    {x_variables} \\geq 0",
            "    \\quad (\\text{integer, production amount in 100kg units})",
            "    $$",
            "",
            "    $$",
            f"    {y_variables} \\in \\{{0,1\\}}",
            "    \\quad (\\text{binary, activation of production line})",
            "    $$",
        ]
    )
    return "\n".join(lines)


def generate_label_model() -> str:
    root = repo_root()
    product_ids, products_table = read_transposed_table(root / clean_path(PRODUCT_DATA_PATH))
    activation_ids, activation_table = read_transposed_table(root / clean_path(ACTIVATION_COST_PATH))
    batch_ids, batch_table = read_transposed_table(root / clean_path(MIN_BATCH_PATH))

    if product_ids != activation_ids or product_ids != batch_ids:
        raise ValueError("Product columns do not match across Mixture9 input files.")

    return build_label_model(
                product_ids,
                products_table,
                activation_table["Activation Cost ($)"],
                batch_table["Minimum Batch Size (100 kg units)"],
            )


def build_lp_model(
    products: list[str],
    product_data: dict[str, dict[str, str]],
    activation_costs: dict[str, str],
    min_batches: dict[str, str],
) -> str:
    demand = product_data["Maximum Demand (100 kg units)"]
    prices = product_data["Selling Price ($/100 kg)"]
    production_costs = product_data["Production Cost ($/100 kg)"]
    daily_quotas = product_data["Production Quota (max per day)"]

    objective: list[str] = []
    for product in products:
        index = product_index(product)
        unit_profit = Decimal(prices[product]) - Decimal(production_costs[product])
        objective.append(f"{display_number(unit_profit)} x_{index}")
        objective.append(f"- {activation_costs[product]} y_{index}")

    lines = ["Maximize", f"    obj: {' + '.join(objective).replace('+ -', '-')}", "Subject To"]
    for product in products:
        index = product_index(product)
        lines.append(f"    demand_{index}: x_{index} <= {demand[product]}")
    for product in products:
        index = product_index(product)
        monthly_quota = int(daily_quotas[product]) * PRODUCTION_DAYS
        lines.append(f"    monthly_quota_{index}: x_{index} <= {monthly_quota}")
    for product in products:
        index = product_index(product)
        lines.append(f"    minimum_batch_{index}: x_{index} - {min_batches[product]} y_{index} >= 0")
    for product in products:
        index = product_index(product)
        lines.append(f"    activation_link_{index}: x_{index} - {demand[product]} y_{index} <= 0")

    x_variables = [f"x_{product_index(product)}" for product in products]
    y_variables = [f"y_{product_index(product)}" for product in products]
    lines.extend(["Generals", "    " + " ".join(x_variables), "Binaries", "    " + " ".join(y_variables), "End"])
    return "\n".join(lines)


def generate_lp_model() -> str:
    root = repo_root()
    product_ids, products_table = read_transposed_table(root / clean_path(PRODUCT_DATA_PATH))
    activation_ids, activation_table = read_transposed_table(root / clean_path(ACTIVATION_COST_PATH))
    batch_ids, batch_table = read_transposed_table(root / clean_path(MIN_BATCH_PATH))
    if product_ids != activation_ids or product_ids != batch_ids:
        raise ValueError("Product columns do not match across Mixture9 input files.")
    return build_lp_model(
        product_ids,
        products_table,
        activation_table["Activation Cost ($)"],
        batch_table["Minimum Batch Size (100 kg units)"],
    )

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
