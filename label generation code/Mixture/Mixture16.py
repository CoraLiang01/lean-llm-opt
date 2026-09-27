#!/usr/bin/env python3
"""Generate the Mixture16 label model from its benchmark data."""

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
    "Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv\n"
    "Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv"
)
OPTION_CHARACTERISTICS_PATH = (
    "Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/OptionCharacteristics.csv"
)
OPTION_ASSET_REFERENCE_MATRIX_PATH = (
    "Test_Dataset/Large-scale-or/Mixture_testing/Mixture16/Option_AssetReferenceMatrix.csv"
)
EXPECTED_LABEL_OBJECTIVE = "6"
PROBLEM_NAME = "Mixture16"
PROBLEM_CATEGORY = "Mixture"
GUROBI_CL_ENV_VAR = "GUROBI_CL"
STDOUT_END = ''


INITIAL_EXPOSURES = {
    "Delta": Decimal("0.25"),
    "Gamma": Decimal("0.08"),
    "Vega": Decimal("0.17"),
}
RISK_TOLERANCES = {
    "Delta": Decimal("0.06"),
    "Gamma": Decimal("0.05"),
    "Vega": Decimal("0.07"),
}
GREEK_LABELS = {
    "Delta": "Delta",
    "Gamma": "Gamma",
    "Vega": "Vega",
}


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path.name} is empty.")
    return rows


def option_number(option_name: str) -> int:
    return int(option_name.split("_", 1)[1])


def six_decimal(value: Decimal | str) -> str:
    return f"{Decimal(str(value)):.6f}"


def asset_reference_count(matrix_row: dict[str, str]) -> int:
    return sum(
        int(value)
        for key, value in matrix_row.items()
        if key.startswith("Asset_")
    )


def signed_expression_term(coefficient: Decimal, variable_name: str, first: bool) -> str:
    if first:
        return f"{six_decimal(coefficient)}*{variable_name}"
    if coefficient < 0:
        return f"- {six_decimal(-coefficient)}*{variable_name}"
    return f"+ {six_decimal(coefficient)}*{variable_name}"


def greek_expression(
    option_rows: list[dict[str, str]],
    matrix_by_option: dict[str, dict[str, str]],
    greek_name: str,
) -> str:
    terms: list[str] = []
    for row in option_rows:
        option = row["Option"]
        coefficient = Decimal(row[greek_name]) * Decimal(asset_reference_count(matrix_by_option[option]))
        if coefficient == 0:
            continue
        variable_name = f"x_{option_number(option)}"
        terms.append(signed_expression_term(coefficient, variable_name, first=not terms))
    return " ".join(terms)


def objective_terms(option_rows: list[dict[str, str]]) -> str:
    terms = [
        f"{six_decimal(row['Cost'])}*z_{option_number(row['Option'])}"
        for row in option_rows
    ]
    return " + ".join(terms)


def build_label_model(
    option_rows: list[dict[str, str]],
    matrix_rows: list[dict[str, str]],
) -> str:
    matrix_key = next(key for key in matrix_rows[0] if key == "" or key is None)
    matrix_by_option = {row[matrix_key]: row for row in matrix_rows}

    lines = [
        "**Variables:** ",
        "    - $x_i$: Integer number of contracts for option $i$, ",
        "    - $z_i$: Auxiliary variable representing $|x_i|$,",
        "",
        "    Variables:",
        "    x_i: Integer (可正负), i = 1..120",
        "    z_i: Continuous, z_i >= 0, i = 1..120",
        "",
        "    Objective:",
        f"    Minimize  Z = {objective_terms(option_rows)}",
        "",
        "    Constraints:",
        "    # Absolute value linearization for z_i = |x_i|",
    ]

    for row in option_rows:
        index = option_number(row["Option"])
        lines.append(f"    z_{index} - x_{index} >= 0")
        lines.append(f"    z_{index} + x_{index} >= 0")

    lines.extend(["", "    # Trading bounds"])
    for row in option_rows:
        index = option_number(row["Option"])
        lines.append(f"    {row['MaxShort']} <= x_{index} <= {row['MaxLong']}")

    for greek_name in ("Delta", "Gamma", "Vega"):
        expression = greek_expression(option_rows, matrix_by_option, greek_name)
        upper_bound = -INITIAL_EXPOSURES[greek_name] + RISK_TOLERANCES[greek_name]
        lower_bound = -INITIAL_EXPOSURES[greek_name] - RISK_TOLERANCES[greek_name]
        lines.extend(
            [
                "",
                f"    # {GREEK_LABELS[greek_name]} band: |{GREEK_LABELS[greek_name]}| <= {RISK_TOLERANCES[greek_name]}",
                f"    {expression} <= {six_decimal(upper_bound)}",
                f"    {expression} >= {six_decimal(lower_bound)}",
            ]
        )

    lines.extend(
        [
            "",
            "    Variable Types:",
            "    x_i ∈ Z, ∀i",
            "    z_i ≥ 0 (continuous), ∀i",
        ]
    )
    return "\n".join(lines)


def generate_label_model() -> str:
    root = repo_root()
    option_rows = read_csv_rows(root / OPTION_CHARACTERISTICS_PATH)
    matrix_rows = read_csv_rows(root / OPTION_ASSET_REFERENCE_MATRIX_PATH)
    return build_label_model(option_rows, matrix_rows)


def build_lp_model(
    option_rows: list[dict[str, str]],
    matrix_rows: list[dict[str, str]],
) -> str:
    matrix_key = next(key for key in matrix_rows[0] if key == "" or key is None)
    matrix_by_option = {row[matrix_key]: row for row in matrix_rows}

    lines = ["Minimize", f"    obj: {objective_terms(option_rows).replace('*', ' ')}", "Subject To"]
    for row in option_rows:
        index = option_number(row["Option"])
        lines.append(f"    abs_pos_{index}: z_{index} - x_{index} >= 0")
        lines.append(f"    abs_neg_{index}: z_{index} + x_{index} >= 0")
    for greek_name in ("Delta", "Gamma", "Vega"):
        expression = greek_expression(option_rows, matrix_by_option, greek_name).replace("*", " ")
        upper_bound = -INITIAL_EXPOSURES[greek_name] + RISK_TOLERANCES[greek_name]
        lower_bound = -INITIAL_EXPOSURES[greek_name] - RISK_TOLERANCES[greek_name]
        lines.append(f"    {greek_name.lower()}_upper: {expression} <= {six_decimal(upper_bound)}")
        lines.append(f"    {greek_name.lower()}_lower: {expression} >= {six_decimal(lower_bound)}")

    lines.append("Bounds")
    for row in option_rows:
        index = option_number(row["Option"])
        lines.append(f"    {row['MaxShort']} <= x_{index} <= {row['MaxLong']}")
        lines.append(f"    z_{index} >= 0")
    lines.extend(["Generals", "    " + " ".join(f"x_{option_number(row['Option'])}" for row in option_rows), "End"])
    return "\n".join(lines)


def generate_lp_model() -> str:
    root = repo_root()
    option_rows = read_csv_rows(root / OPTION_CHARACTERISTICS_PATH)
    matrix_rows = read_csv_rows(root / OPTION_ASSET_REFERENCE_MATRIX_PATH)
    return build_lp_model(option_rows, matrix_rows)

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
