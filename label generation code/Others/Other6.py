#!/usr/bin/env python3
"""Generate the Other6 label model from its benchmark data."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import csv
from decimal import Decimal
from pathlib import Path


DATASET_ADDRESS = "Test_Dataset/Large-scale-or/Others_example/Others6/18.csv"
EXPECTED_LABEL_OBJECTIVE = "22.85"
PROBLEM_NAME = "Other6"
PROBLEM_CATEGORY = "Others"
GUROBI_CL_ENV_VAR = "GUROBI_CL"
STDOUT_END = '\n'

CPU_FREQUENCIES_GHZ = ("1.33", "2", "2.66")


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


def read_basic_instructions(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("18.csv is empty.")

    row = rows[0]
    task_columns = [column for column in row.keys() if column != "Process"]
    task_columns.sort(key=int)
    return [row[column] for column in task_columns]


def display_instruction(value: str) -> str:
    return f"{Decimal(value):.1f}"


def build_assignment_constraint(task_number: int) -> str:
    terms = [f"x_{{{cpu},{task_number}}}" for cpu in range(1, len(CPU_FREQUENCIES_GHZ) + 1)]
    expression = " + ".join(terms)
    return (
        f"    $${expression} = 1 \\quad "
        f"(\\text{{task {task_number} assigned}})$$"
    )


def build_cpu_time_constraint(
    cpu_number: int,
    frequency: str,
    basic_instructions: list[str],
) -> str:
    terms = [
        f"\\frac{{{display_instruction(value)}}}{{{frequency}}}x_{{{cpu_number},{task_number}}}"
        for task_number, value in enumerate(basic_instructions, start=1)
    ]
    expression = " + ".join(terms)
    return (
        "    $$\n"
        f"    {expression} \\leq C_{{\\max}} \\quad "
        f"(\\text{{CPU {cpu_number} time}})\n"
        "    $$"
    )


def build_label_model(basic_instructions: list[str]) -> str:
    task_count = len(basic_instructions)
    lines = [
        "",
        "",
        "    ## Mathematical Model",
        "",
        "    **Minimize**",
        "",
        "    $$C_{\\max}$$",
        "",
        "    **subject to**",
        "",
    ]

    for task_number in range(1, task_count + 1):
        lines.append(build_assignment_constraint(task_number))

    lines.extend(["", "    **CPU time constraints**", ""])

    for cpu_number, frequency in enumerate(CPU_FREQUENCIES_GHZ, start=1):
        lines.append(build_cpu_time_constraint(cpu_number, frequency, basic_instructions))
        lines.append("")

    lines.extend(
        [
            "    **where**",
            "",
            (
                "    $$x_{i,j} \\in \\{0,1\\} \\quad "
                f"(i=1,2,3; \\ j=1,...,{task_count}; \\ "
                "\\text{binary, indicates if task } j \\text{ is assigned to CPU } i)$$"
            ),
            "",
            "    $$C_{\\max} \\geq 0 \\quad (\\text{continuous, completion time})$$",
        ]
    )
    return "\n".join(lines)


def generate_label_model() -> str:
    instruction_list = read_basic_instructions(data_path())
    return build_label_model(instruction_list)


def lp_expression(terms: list[tuple[str, str]]) -> str:
    return " + ".join(f"{coef} {var}" if coef != "1" else var for coef, var in terms)


def lp_decimal(value: Decimal) -> str:
    return f"{value:.12g}"


def build_lp_model(basic_instructions: list[str]) -> str:
    task_count = len(basic_instructions)
    cpu_count = len(CPU_FREQUENCIES_GHZ)
    lines = ["Minimize", "    obj: Cmax", "Subject To"]

    for task_number in range(1, task_count + 1):
        terms = [("1", f"x_{cpu}_{task_number}") for cpu in range(1, cpu_count + 1)]
        lines.append(f"    assign_task_{task_number}: {lp_expression(terms)} = 1")

    for cpu_number, frequency in enumerate(CPU_FREQUENCIES_GHZ, start=1):
        terms = [
            (lp_decimal(Decimal(value) / Decimal(frequency)), f"x_{cpu_number}_{task_number}")
            for task_number, value in enumerate(basic_instructions, start=1)
        ]
        expression = lp_expression(terms)
        lines.append(f"    cpu_time_{cpu_number}: {expression} - Cmax <= 0")

    binaries = [f"x_{cpu}_{task}" for cpu in range(1, cpu_count + 1) for task in range(1, task_count + 1)]
    lines.extend(["Bounds", "    Cmax >= 0", "Binaries", "    " + " ".join(binaries), "End"])
    return "\n".join(lines)


def generate_lp_model() -> str:
    instruction_list = read_basic_instructions(data_path())
    return build_lp_model(instruction_list)

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
