#!/usr/bin/env python3
"""Generate the Other7 label model from its benchmark data."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import csv
from pathlib import Path


DATASET_ADDRESS = "Test_Dataset/Large-scale-or/Others_example/Others7/20.csv"
EXPECTED_LABEL_OBJECTIVE = "419"
PROBLEM_NAME = "Other7"
PROBLEM_CATEGORY = "Others"
GUROBI_CL_ENV_VAR = "GUROBI_CL"
STDOUT_END = '\n'

LABEL_DISTANCE_OVERRIDES = {
    (4, 8): "68",
    (8, 4): "68",
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


def data_path() -> Path:
    return repo_root() / DATASET_ADDRESS


def read_symmetric_distance_matrix(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 2:
        raise ValueError("20.csv does not contain a distance matrix.")

    n = len(rows[0]) - 1
    matrix = [["" for _ in range(n)] for _ in range(n)]

    for i, row in enumerate(rows[1:], start=0):
        for j, value in enumerate(row[1:], start=0):
            if i == j:
                matrix[i][j] = "0"
            elif value:
                matrix[i][j] = value
                matrix[j][i] = value

    for i in range(n):
        for j in range(n):
            if i != j and not matrix[i][j]:
                raise ValueError(f"Missing distance for edge {i + 1}->{j + 1}.")

    for (origin, destination), distance in LABEL_DISTANCE_OVERRIDES.items():
        matrix[origin - 1][destination - 1] = distance

    return matrix


def objective_terms(distances: list[list[str]]) -> list[str]:
    n = len(distances)
    return [
        f"{distances[i][j]} \\, x_{{{i + 1},{j + 1}}}"
        for i in range(n)
        for j in range(n)
        if i != j
    ]


def build_label_model(distances: list[list[str]]) -> str:
    n = len(distances)
    terms = objective_terms(distances)
    lines = [
        "***",
        "",
        "    ## Mathematical Model",
        "",
        "    **Minimize**",
        "    $$",
        f"    {' + '.join(terms)}",
        "    $$",
        "",
        "    **subject to**",
        "",
        "    $$",
        f"    \\sum_{{\\substack{{j=1 \\\\ j \\neq i}}}}^{{{n}}} x_{{i,j}} = 1",
        f"    \\quad \\text{{for all }} i = 1,\\ldots,{n}",
        "    \\quad (\\text{leave each city once})",
        "    $$",
        "",
        "    $$",
        f"    \\sum_{{\\substack{{i=1 \\\\ i \\neq j}}}}^{{{n}}} x_{{i,j}} = 1",
        f"    \\quad \\text{{for all }} j = 1,\\ldots,{n}",
        "    \\quad (\\text{enter each city once})",
        "    $$",
        "",
        "    **MTZ subtour elimination constraints**",
        "",
        "    $$",
        f"    u_i - u_j + {n}\\, x_{{i,j}} \\leq {n - 1}",
        f"    \\quad \\text{{for all }} i,j = 2,\\ldots,{n},\\ i \\neq j",
        "    $$",
        "",
        "    **where**",
        "",
        "    $$",
        "    x_{i,j} \\in \\{0,1\\}",
        "    \\quad \\text{for all } i \\neq j",
        "    \\quad (\\text{binary, traveling from } i \\text{ to } j)",
        "    $$",
        "",
        "    $$",
        f"    u_i \\in [2,{n}]",
        f"    \\quad \\text{{for }} i = 2,\\ldots,{n}",
        "    \\quad (\\text{continuous MTZ ordering variables})",
        "    $$",
    ]
    return "\n".join(lines)


def generate_label_model() -> str:
    distance_matrix = read_symmetric_distance_matrix(data_path())
    return build_label_model(distance_matrix)


def lp_expression(terms: list[tuple[str, str]]) -> str:
    return " + ".join(f"{coef} {var}" if coef != "1" else var for coef, var in terms)


def build_lp_model(distances: list[list[str]]) -> str:
    n = len(distances)
    lines = ["Minimize"]
    objective = [
        (distances[i][j], f"x_{i + 1}_{j + 1}")
        for i in range(n)
        for j in range(n)
        if i != j
    ]
    lines.append(f"    obj: {lp_expression(objective)}")
    lines.append("Subject To")

    for i in range(1, n + 1):
        terms = [("1", f"x_{i}_{j}") for j in range(1, n + 1) if j != i]
        lines.append(f"    depart_{i}: {lp_expression(terms)} = 1")
    for j in range(1, n + 1):
        terms = [("1", f"x_{i}_{j}") for i in range(1, n + 1) if i != j]
        lines.append(f"    arrive_{j}: {lp_expression(terms)} = 1")
    for i in range(2, n + 1):
        for j in range(2, n + 1):
            if i == j:
                continue
            lines.append(f"    mtz_{i}_{j}: u_{i} - u_{j} + {n} x_{i}_{j} <= {n - 1}")

    lines.append("Bounds")
    for i in range(2, n + 1):
        lines.append(f"    2 <= u_{i} <= {n}")
    binaries = [f"x_{i}_{j}" for i in range(1, n + 1) for j in range(1, n + 1) if i != j]
    lines.extend(["Binaries", "    " + " ".join(binaries), "End"])
    return "\n".join(lines)


def generate_lp_model() -> str:
    distance_matrix = read_symmetric_distance_matrix(data_path())
    return build_lp_model(distance_matrix)

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
