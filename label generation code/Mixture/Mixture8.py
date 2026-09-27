#!/usr/bin/env python3
"""Generate the Mixture8 label model from its benchmark data."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import csv
import re
from decimal import Decimal
from pathlib import Path


DATASET_ADDRESS = (
    "Test_Dataset/Large-scale-or//Mixture_testing/Mixture8/30-1.csv\n"
    "Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv"
)
RAW_MATERIAL_PATH = "Test_Dataset/Large-scale-or//Mixture_testing/Mixture8/30-1.csv"
BRAND_PATH = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv"
EXPECTED_LABEL_OBJECTIVE = "4500"
PROBLEM_NAME = "Mixture8"
PROBLEM_CATEGORY = "Mixture"
GUROBI_CL_ENV_VAR = "GUROBI_CL"
STDOUT_END = ''

MIN_RED_PRODUCTION = "2000"


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


def read_raw_materials(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("30-1.csv is empty.")

    required_columns = {"Grade", "Daily Supply (kg)", "Cost (CNY/kg)"}
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return rows


def read_brands(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("30-2.csv is empty.")

    required_columns = {"Brand", "Blending Requirements", "Selling Price (CNY/kg)"}
    missing_columns = required_columns.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return rows


def signed_term(coefficient: Decimal, variable: str, first: bool = False) -> str:
    sign = "-" if coefficient < 0 else "+"
    magnitude = abs(coefficient)
    if magnitude == Decimal(1):
        body = variable
    else:
        body = f"{display_number(magnitude)} {variable}"

    if first:
        return f"- {body}" if sign == "-" else body
    return f"{sign} {body}"


def objective_lines(terms: list[tuple[Decimal, str]]) -> list[str]:
    lines: list[str] = []
    for start in range(0, len(terms), 4):
        chunk = terms[start : start + 4]
        indent = "  " if start == 0 else "   "
        pieces = [
            signed_term(coefficient, variable, first=(start == 0 and index == 0))
            for index, (coefficient, variable) in enumerate(chunk)
        ]
        lines.append(f"{indent}{' '.join(pieces)}")
    return lines


def parse_requirements(requirement_text: str) -> list[tuple[str, str, Decimal]]:
    pattern = re.compile(r"(I{1,3})\s+(less than|more than)\s+([0-9]+)%")
    requirements: list[tuple[str, str, Decimal]] = []
    for grade, relation, percent in pattern.findall(requirement_text):
        requirements.append((grade, relation, Decimal(percent) / Decimal(100)))
    return requirements


def ratio_terms(brand: str, grades: list[str], selected_grade: str, ratio: Decimal) -> list[tuple[Decimal, str]]:
    terms: list[tuple[Decimal, str]] = []
    for grade in grades:
        coefficient = Decimal(1) - ratio if grade == selected_grade else -ratio
        terms.append((coefficient, f"x[{brand},{grade}]"))
    return terms


def expression(terms: list[tuple[Decimal, str]]) -> str:
    pieces = [
        signed_term(coefficient, variable, first=(index == 0))
        for index, (coefficient, variable) in enumerate(terms)
    ]
    return " ".join(pieces)


def continuation_expression(terms: list[tuple[Decimal, str]]) -> str:
    return " ".join(signed_term(coefficient, variable) for coefficient, variable in terms)


def build_label_model(raw_materials: list[dict[str, str]], brands: list[dict[str, str]]) -> str:
    grades = [row["Grade"] for row in raw_materials]
    costs = {row["Grade"]: Decimal(row["Cost (CNY/kg)"]) for row in raw_materials}

    objective_terms_list: list[tuple[Decimal, str]] = []
    for brand_row in brands:
        brand = brand_row["Brand"]
        selling_price = Decimal(brand_row["Selling Price (CNY/kg)"])
        for grade in grades:
            objective_terms_list.append((selling_price - costs[grade], f"x[{brand},{grade}]"))

    lines = [
        r"\ Model Mixture8_LabelModel",
        r"\ LP format - for model browsing. Use MPS format to capture full model detail.",
        "Maximize",
    ]
    lines.extend(objective_lines(objective_terms_list))
    lines.append("Subject To")

    for raw_row in raw_materials:
        grade = raw_row["Grade"]
        supply_terms = " + ".join(f"x[{brand_row['Brand']},{grade}]" for brand_row in brands)
        lines.append(
            f" supply_{grade}: {supply_terms} <= {display_number(raw_row['Daily Supply (kg)'])}"
        )

    for brand_row in brands:
        brand = brand_row["Brand"]
        for grade, relation, ratio in parse_requirements(brand_row["Blending Requirements"]):
            relation_name = "upper" if relation == "less than" else "lower"
            sense = "<=" if relation == "less than" else ">="
            label = f" {brand}_{grade}_{relation_name}_ratio:"
            terms = ratio_terms(brand, grades, grade, ratio)
            if brand == "Yellow" and grade in {"I", "III"}:
                first_line_terms = terms[:2]
                second_line_terms = terms[2:]
                lines.append(f"{label} {expression(first_line_terms)}")
                lines.append(f"   {continuation_expression(second_line_terms)} {sense} 0")
            elif brand == "Blue" and grade == "II":
                lines.append(f"{label} {expression(terms)}")
                lines.append(f"   {sense} 0")
            else:
                lines.append(f"{label} {expression(terms)} {sense} 0")

    red_terms = " + ".join(f"x[Red,{grade}]" for grade in grades)
    lines.extend(
        [
            f" min_production_Red: {red_terms} >= {MIN_RED_PRODUCTION}",
            "Bounds",
            "End",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_label_model() -> str:
    root = repo_root()
    raw_material_rows = read_raw_materials(root / clean_path(RAW_MATERIAL_PATH))
    brand_rows = read_brands(root / clean_path(BRAND_PATH))
    return build_label_model(raw_material_rows, brand_rows)

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
