#!/usr/bin/env python3
"""Generate the Mixture3 label model from its benchmark data."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import csv
from decimal import Decimal
from pathlib import Path


DATASET_ADDRESS = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture3/43.csv"
EXPECTED_LABEL_OBJECTIVE = "1146.57"
PROBLEM_NAME = "Mixture3"
PROBLEM_CATEGORY = "Mixture"
GUROBI_CL_ENV_VAR = "GUROBI_CL"
STDOUT_END = '\n'

PROCEDURE_A_EQUIPMENT = ("A1", "A2")
PROCEDURE_B_EQUIPMENT = ("B1", "B2", "B3")
PRODUCTS = ("I", "II", "III")
VARIABLES = (
    ("A1", "I"),
    ("A1", "II"),
    ("A2", "I"),
    ("A2", "II"),
    ("A2", "III"),
    ("B1", "I"),
    ("B1", "II"),
    ("B2", "I"),
    ("B2", "III"),
    ("B3", "I"),
)


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


def display_number(value: str | Decimal, force_decimal_for_integer: bool = False) -> str:
    number = Decimal(str(value))
    if force_decimal_for_integer and number == number.to_integral_value():
        return f"{number.quantize(Decimal(1))}.0"
    if number == number.to_integral_value():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def read_table(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    equipment: dict[str, dict[str, str]] = {}
    raw_cost: dict[str, str] = {}
    unit_price: dict[str, str] = {}

    for row in rows:
        key = row["Equipment / Cost"]
        if key == "Raw Material Cost (yuan/unit)":
            raw_cost = {product: row[f"Product {product}"] for product in PRODUCTS}
        elif key == "Unit Price (yuan/unit)":
            unit_price = {product: row[f"Product {product}"] for product in PRODUCTS}
        elif key in {*PROCEDURE_A_EQUIPMENT, *PROCEDURE_B_EQUIPMENT}:
            equipment[key] = row

    return equipment, raw_cost, unit_price


def time_term(equipment: dict[str, dict[str, str]], machine: str, products: tuple[str, ...]) -> str:
    terms = [
        f"{display_number(equipment[machine][f'Product {product}'])}x_{{{machine},{product}}}"
        for product in products
    ]
    return " + ".join(terms)


def build_label_model(
    equipment: dict[str, dict[str, str]],
    raw_cost: dict[str, str],
    unit_price: dict[str, str],
) -> str:
    lines = [
        "### Linear Programming Formulation: Optimal Production Plan",
        "",
        "    **Decision Variables:**",
        "    - $x_{A1,I}$: Number of units of Product I processed on A1 (procedure A)",
        "    - $x_{A1,II}$: Number of units of Product II processed on A1 (procedure A)",
        "    - $x_{A2,I}$: Number of units of Product I processed on A2 (procedure A)",
        "    - $x_{A2,II}$: Number of units of Product II processed on A2 (procedure A)",
        "    - $x_{A2,III}$: Number of units of Product III processed on A2 (procedure A)",
        "    - $x_{B1,I}$: Number of units of Product I processed on B1 (procedure B)",
        "    - $x_{B1,II}$: Number of units of Product II processed on B1 (procedure B)",
        "    - $x_{B2,I}$: Number of units of Product I processed on B2 (procedure B)",
        "    - $x_{B2,III}$: Number of units of Product III processed on B2 (procedure B)",
        "    - $x_{B3,I}$: Number of units of Product I processed on B3 (procedure B)",
        "",
        "    All variables are continuous and non-negative ($\\geq 0$).",
        "",
        "    ---",
        "",
        "    **Objective Function (Maximize Profit):**",
        "",
        "    \\[",
        "    \\max \\quad",
        "    \\Big[",
        f"    {display_number(unit_price['I'], True)}(x_{{A1,I}} + x_{{A2,I}}) + {display_number(unit_price['II'], True)}(x_{{A1,II}} + x_{{A2,II}}) + {display_number(unit_price['III'], True)}(x_{{A2,III}})",
        f"    - {display_number(raw_cost['I'])}(x_{{A1,I}} + x_{{A2,I}}) - {display_number(raw_cost['II'])}(x_{{A1,II}} + x_{{A2,II}}) - {display_number(raw_cost['III'])}(x_{{A2,III}})",
        "    \\Big]",
        "    - \\Bigg[",
        f"    \\frac{{{display_number(equipment['A1']['Equipment Cost at Full Load (yuan)'])}}}{{{display_number(equipment['A1']['Available Equipment Operating Time'])}}}({time_term(equipment, 'A1', ('I', 'II'))})",
        f"    + \\frac{{{display_number(equipment['A2']['Equipment Cost at Full Load (yuan)'])}}}{{{display_number(equipment['A2']['Available Equipment Operating Time'])}}}({time_term(equipment, 'A2', ('I', 'II', 'III'))})",
        f"    + \\frac{{{display_number(equipment['B1']['Equipment Cost at Full Load (yuan)'])}}}{{{display_number(equipment['B1']['Available Equipment Operating Time'])}}}({time_term(equipment, 'B1', ('I', 'II'))})",
        f"    + \\frac{{{display_number(equipment['B2']['Equipment Cost at Full Load (yuan)'])}}}{{{display_number(equipment['B2']['Available Equipment Operating Time'])}}}({time_term(equipment, 'B2', ('I', 'III'))})",
        f"    + \\frac{{{display_number(equipment['B3']['Equipment Cost at Full Load (yuan)'])}}}{{{display_number(equipment['B3']['Available Equipment Operating Time'])}}}({time_term(equipment, 'B3', ('I',))})",
        "    \\Bigg]",
        "    \\]",
        "",
        "    ---",
        "",
        "    **Constraints:**",
        "",
        "    **1. Procedure A Equipment Capacity Constraints:**",
        "    \\[",
        f"    {time_term(equipment, 'A1', ('I', 'II'))} \\leq {display_number(equipment['A1']['Available Equipment Operating Time'])}",
        "    \\]",
        "    \\[",
        f"    {time_term(equipment, 'A2', ('I', 'II', 'III'))} \\leq {display_number(equipment['A2']['Available Equipment Operating Time'])}",
        "    \\]",
        "",
        "    **2. Procedure B Equipment Capacity Constraints:**",
        "    \\[",
        f"    {time_term(equipment, 'B1', ('I', 'II'))} \\leq {display_number(equipment['B1']['Available Equipment Operating Time'])}",
        "    \\]",
        "    \\[",
        f"    {time_term(equipment, 'B2', ('I', 'III'))} \\leq {display_number(equipment['B2']['Available Equipment Operating Time'])}",
        "    \\]",
        "    \\[",
        f"    {time_term(equipment, 'B3', ('I',))} \\leq {display_number(equipment['B3']['Available Equipment Operating Time'])}",
        "    \\]",
        "",
        "    **3. Product Flow Balance Constraints:**",
        "    - For Product I:",
        "    \\[",
        "    x_{A1,I} + x_{A2,I} = x_{B1,I} + x_{B2,I} + x_{B3,I}",
        "    \\]",
        "    - For Product II:",
        "    \\[",
        "    x_{A1,II} + x_{A2,II} = x_{B1,II}",
        "    \\]",
        "    - For Product III:",
        "    \\[",
        "    x_{A2,III} = x_{B2,III}",
        "    \\]",
        "",
        "    **4. Equipment Eligibility Constraints:**",
        "    - $x_{A1,III} = 0$",
        "    - $x_{B1,III} = 0$",
        "    - $x_{B2,II} = 0$",
        "    - $x_{B3,II} = 0$",
        "    - $x_{B3,III} = 0$",
        "    - $x_{A1,III} = 0$",
        "",
        "    **5. Non-negativity Constraints:**",
        "    \\[",
        "    x_{A1,I} \\geq 0,\\quad x_{A1,II} \\geq 0,\\quad x_{A2,I} \\geq 0,\\quad x_{A2,II} \\geq 0,\\quad x_{A2,III} \\geq 0,",
        "    \\]",
        "    \\[",
        "    x_{B1,I} \\geq 0,\\quad x_{B1,II} \\geq 0,\\quad x_{B2,I} \\geq 0,\\quad x_{B2,III} \\geq 0,\\quad x_{B3,I} \\geq 0",
        "    \\]",
    ]
    return "\n".join(lines)


def generate_label_model() -> str:
    equipment_data, raw_material_cost, price = read_table(repo_root() / DATASET_ADDRESS)
    return build_label_model(equipment_data, raw_material_cost, price)


def lp_number(value: Decimal | str) -> str:
    return display_number(value)


def signed_lp_expression(terms: list[tuple[Decimal | str, str]]) -> str:
    pieces: list[str] = []
    for coefficient, variable in terms:
        value = Decimal(str(coefficient))
        if value == 0:
            continue
        magnitude = lp_number(abs(value))
        body = variable if magnitude == "1" else f"{magnitude} {variable}"
        if not pieces:
            pieces.append(f"- {body}" if value < 0 else body)
        else:
            pieces.append(f"- {body}" if value < 0 else f"+ {body}")
    return " ".join(pieces) if pieces else "0"


def equipment_rate(equipment: dict[str, dict[str, str]], machine: str) -> Decimal:
    return Decimal(equipment[machine]["Equipment Cost at Full Load (yuan)"]) / Decimal(
        equipment[machine]["Available Equipment Operating Time"]
    )


def build_lp_model(
    equipment: dict[str, dict[str, str]],
    raw_cost: dict[str, str],
    unit_price: dict[str, str],
) -> str:
    objective: list[tuple[Decimal | str, str]] = []
    for machine, product in VARIABLES:
        coefficient = -equipment_rate(equipment, machine) * Decimal(equipment[machine][f"Product {product}"])
        if machine.startswith("A"):
            coefficient += Decimal(unit_price[product]) - Decimal(raw_cost[product])
        objective.append((coefficient, f"x_{machine}_{product}"))

    lines = ["Maximize", f"    obj: {signed_lp_expression(objective)}", "Subject To"]
    machine_products = {
        "A1": ("I", "II"),
        "A2": ("I", "II", "III"),
        "B1": ("I", "II"),
        "B2": ("I", "III"),
        "B3": ("I",),
    }
    for machine, products in machine_products.items():
        terms = [
            (equipment[machine][f"Product {product}"], f"x_{machine}_{product}")
            for product in products
        ]
        rhs = equipment[machine]["Available Equipment Operating Time"]
        lines.append(f"    capacity_{machine}: {signed_lp_expression(terms)} <= {rhs}")

    lines.extend(
        [
            "    flow_I: x_A1_I + x_A2_I - x_B1_I - x_B2_I - x_B3_I = 0",
            "    flow_II: x_A1_II + x_A2_II - x_B1_II = 0",
            "    flow_III: x_A2_III - x_B2_III = 0",
            "End",
        ]
    )
    return "\n".join(lines)


def generate_lp_model() -> str:
    equipment_data, raw_material_cost, price = read_table(repo_root() / DATASET_ADDRESS)
    return build_lp_model(equipment_data, raw_material_cost, price)

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
