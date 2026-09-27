#!/usr/bin/env python3
"""Generate the Mixture15 label model from its benchmark data."""

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
    "Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv\n"
    "Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv\n"
    "Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv"
)
SCHOOL_CAPACITY_PATH = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv"
NEIGHBORHOODS_PATH = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv"
DISTANCE_PATH = "Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv"
EXPECTED_LABEL_OBJECTIVE = "5044"
PROBLEM_NAME = "Mixture15"
PROBLEM_CATEGORY = "Mixture"
GUROBI_CL_ENV_VAR = "GUROBI_CL"
STDOUT_END = ''

WHITE_MIN_COEFFICIENTS = ("0.5", "-0.5")
WHITE_MAX_COEFFICIENTS = ("0.3", "-0.7")


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
    if number.as_tuple().exponent < -2:
        number = number.quantize(Decimal("0.01"))
    if number == number.to_integral_value():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path.name} is empty.")
    return rows


def distance_by_school(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        row["School"]: {
            key: display_number(value)
            for key, value in row.items()
            if key != "School"
        }
        for row in rows
    }


def capacity_by_school(rows: list[dict[str, str]]) -> dict[str, str]:
    return {row["School"]: display_number(row["Capacity"]) for row in rows}


def variable(kind: str, school: str, neighborhood: str) -> str:
    return f"x_{kind}_{school}_{neighborhood}"


def signed_term(coefficient: str, var_name: str, first: bool = False) -> str:
    value = Decimal(coefficient)
    magnitude = abs(value)
    body = f"{display_number(magnitude)} {var_name}"
    if first:
        return f"- {body}" if value < 0 else body
    sign = "-" if value < 0 else "+"
    return f"{sign} {body}"


def objective_terms(
    schools: list[str],
    neighborhoods: list[str],
    distances: dict[str, dict[str, str]],
) -> list[str]:
    terms: list[str] = []
    for school in schools:
        for neighborhood in neighborhoods:
            distance = distances[school][neighborhood]
            terms.append(f"{distance} {variable('w', school, neighborhood)}")
            terms.append(f"{distance} {variable('n', school, neighborhood)}")
    return terms


def append_chunked_sum(lines: list[str], terms: list[str], chunk_size: int) -> None:
    for start in range(0, len(terms), chunk_size):
        chunk = terms[start : start + chunk_size]
        prefix = "" if start == 0 else "+ "
        lines.append(f"{prefix}{' + '.join(chunk)}")


def wrap_by_width(
    label: str,
    terms: list[str],
    sense_suffix: str,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    current = label
    for index, term in enumerate(terms):
        piece = term if index == 0 else f"+ {term}"
        separator = "" if current == label else " "
        if current != label and len(current) + len(separator) + len(piece) > max_width:
            lines.append(current)
            current = f"+ {term}"
        else:
            current = f"{current}{separator}{piece}"
    lines.append(f"{current} {sense_suffix}")
    return lines


def wrap_by_chunks(label: str, terms: list[str], sense_suffix: str, first_count: int, later_count: int) -> list[str]:
    lines: list[str] = []
    start = 0
    count = first_count
    while start < len(terms):
        chunk = terms[start : start + count]
        prefix = label if start == 0 else "+ "
        lines.append(f"{prefix}{' + '.join(chunk)}")
        start += count
        count = later_count
    lines[-1] = f"{lines[-1]} {sense_suffix}"
    return lines


def balance_terms(school: str, neighborhoods: list[str], coefficients: tuple[str, str]) -> list[str]:
    white_coefficient, nonwhite_coefficient = coefficients
    terms: list[str] = []
    for neighborhood in neighborhoods:
        terms.append(signed_term(white_coefficient, variable("w", school, neighborhood), first=not terms))
        terms.append(signed_term(nonwhite_coefficient, variable("n", school, neighborhood)))
    return terms


def wrap_signed_terms(label: str, terms: list[str], sense_suffix: str) -> list[str]:
    lines: list[str] = []
    start = 0
    count = 3
    while start < len(terms):
        chunk = terms[start : start + count]
        prefix = label if start == 0 else ""
        lines.append(f"{prefix}{' '.join(chunk)}")
        start += count
        count = 4
    lines[-1] = f"{lines[-1]} {sense_suffix}"
    return lines


def all_variables(schools: list[str], neighborhoods: list[str]) -> list[str]:
    names: list[str] = []
    for school in schools:
        for neighborhood in neighborhoods:
            names.append(variable("w", school, neighborhood))
            names.append(variable("n", school, neighborhood))
    return names


def wrap_items_by_width(items: list[str], max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for item in items:
        if not current:
            current = item
        elif len(current) + 1 + len(item) <= max_width:
            current = f"{current} {item}"
        else:
            lines.append(current)
            current = item
    if current:
        lines.append(current)
    return lines


def build_label_model(
    capacities: dict[str, str],
    neighborhoods_rows: list[dict[str, str]],
    distances: dict[str, dict[str, str]],
) -> str:
    schools = list(capacities.keys())
    neighborhoods = [row["Neighborhood"] for row in neighborhoods_rows]

    lines = ["", "    Minimize"]
    append_chunked_sum(lines, objective_terms(schools, neighborhoods, distances), 4)
    lines.append("    Subject To")

    for row in neighborhoods_rows:
        neighborhood = row["Neighborhood"]
        lines.append(
            f"white_supply[{neighborhood}]: "
            f"{variable('w', schools[0], neighborhood)} + {variable('w', schools[1], neighborhood)} = "
            f"{display_number(row['Population_White'])}"
        )
        lines.append(
            f"nonwhite_supply[{neighborhood}]: "
            f"{variable('n', schools[0], neighborhood)} + {variable('n', schools[1], neighborhood)} = "
            f"{display_number(row['Population_NonWhite'])}"
        )

    for school in schools:
        terms = []
        for neighborhood in neighborhoods:
            terms.append(variable("w", school, neighborhood))
            terms.append(variable("n", school, neighborhood))
        first_count = 5 if school == "I" else 4
        later_count = 6 if school == "I" else 5
        lines.extend(wrap_by_chunks(f"capacity[{school}]: ", terms, f"<= {capacities[school]}", first_count, later_count))

    for school in schools:
        lines.extend(
            wrap_signed_terms(
                f"white_min[{school}]: ",
                balance_terms(school, neighborhoods, WHITE_MIN_COEFFICIENTS),
                ">= 0",
            )
        )
        lines.extend(
            wrap_signed_terms(
                f"white_max[{school}]: ",
                balance_terms(school, neighborhoods, WHITE_MAX_COEFFICIENTS),
                "<= 0",
            )
        )

    lines.extend(["    Bounds", "    Generals"])
    variables = all_variables(schools, neighborhoods)
    lines.extend(wrap_items_by_width(variables, 70))
    lines.append("    End")
    return "\n".join(lines) + "\n"


def generate_label_model() -> str:
    root = repo_root()
    school_capacities = capacity_by_school(read_csv_rows(root / SCHOOL_CAPACITY_PATH))
    neighborhood_data = read_csv_rows(root / NEIGHBORHOODS_PATH)
    school_distances = distance_by_school(read_csv_rows(root / DISTANCE_PATH))
    return build_label_model(school_capacities, neighborhood_data, school_distances)

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
