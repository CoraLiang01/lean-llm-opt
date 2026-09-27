#!/usr/bin/env python3
"""Generate the Mixture7 label model from its benchmark data."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import csv
from decimal import Decimal
from pathlib import Path


DATASET_ADDRESS = "Test_Dataset/Large-scale-or//Mixture_testing/Mixture7/15.csv"
EXPECTED_LABEL_OBJECTIVE = "34"
PROBLEM_NAME = "Mixture7"
PROBLEM_CATEGORY = "Mixture"
GUROBI_CL_ENV_VAR = "GUROBI_CL"
STDOUT_END = ''

WORKERS_TO_SELECT = 10


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


def dataset_path(root: Path) -> Path:
    return root / DATASET_ADDRESS.replace("//", "/")


def display_number(value: str | Decimal) -> str:
    number = Decimal(str(value))
    if number == number.to_integral_value():
        return str(number.quantize(Decimal(1)))
    return format(number.normalize(), "f")


def read_task_times(path: Path) -> tuple[list[int], list[str], dict[int, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("15.csv is empty.")

    tasks = [column for column in rows[0] if column != "Task Time Required"]
    workers: list[int] = []
    task_times: dict[int, dict[str, str]] = {}

    for row in rows:
        worker_value = row["Task Time Required"].strip()
        if not worker_value or not worker_value.isdigit():
            continue
        worker = int(worker_value)
        workers.append(worker)
        task_times[worker] = {task: display_number(row[task]) for task in tasks}

    if not workers:
        raise ValueError("No worker rows found in 15.csv.")

    return workers, tasks, task_times


def worker_task_terms(worker: int, tasks: list[str], coefficients: dict[str, str] | None = None) -> str:
    if coefficients is None:
        terms = [f"x_{worker}{task}" for task in tasks]
    else:
        terms = [f"{coefficients[task]}x_{worker}{task}" for task in tasks]
    return " + ".join(terms)


def task_worker_terms(task: str, workers: list[int]) -> str:
    return " + ".join(f"x_{worker}{task}" for worker in workers)


def worker_set(workers: list[int]) -> str:
    return "{" + ",".join(str(worker) for worker in workers) + "}"


def task_set(tasks: list[str]) -> str:
    return "{" + ",".join(tasks) + "}"


def build_label_model(
    workers: list[int],
    tasks: list[str],
    task_times: dict[int, dict[str, str]],
) -> str:
    lines = [
        "Mathematical Formulation for the Worker–Task Assignment Problem",
        "",
        "Sets",
        "Workers:",
        f"I = {worker_set(workers)}",
        "",
        "Tasks:",
        f"J = {task_set(tasks)}",
        "",
        "Decision Variables",
        "For each worker i in I and task j in J,",
        "",
        "x_ij = 1  if worker i is assigned to task j",
        "x_ij = 0  otherwise",
        "",
        "Thus,",
        "x_ij in {0,1}    for all i in I, j in J",
        "",
        "Objective Function",
        "Minimize total working hours:",
        "",
        "min Z =",
    ]

    for index, worker in enumerate(workers):
        prefix = "" if index == 0 else "+ "
        lines.append(f"{prefix}{worker_task_terms(worker, tasks, task_times[worker])}")

    lines.extend(
        [
            "",
            "Subject to",
            "",
            "1. Each task must be assigned to exactly one worker",
            "",
        ]
    )

    for task in tasks:
        lines.append(f"{task_worker_terms(task, workers)} = 1")

    lines.extend(
        [
            "",
            "2. Each worker can be assigned to at most one task",
            "",
        ]
    )

    for worker in workers:
        lines.append(f"{worker_task_terms(worker, tasks)} <= 1")

    lines.extend(
        [
            "",
            f"3. Exactly {WORKERS_TO_SELECT} workers are selected / exactly {WORKERS_TO_SELECT} assignments are made",
            "",
            f"sum_{{i=1}}^{{{len(workers)}}} sum_{{j in {{A,...,J}}}} x_ij = {WORKERS_TO_SELECT}",
            "",
            "Equivalently,",
        ]
    )

    for index, worker in enumerate(workers):
        prefix = "" if index == 0 else "+ "
        lines.append(f"{prefix}{worker_task_terms(worker, tasks)}")

    lines.extend(
        [
            f"= {WORKERS_TO_SELECT}",
            "",
            "4. Binary restrictions",
            "",
            f"x_ij in {{0,1}}   for all i = 1,...,{len(workers)} and j in {task_set(tasks)}",
        ]
    )

    return "\n".join(lines) + "\n"


def generate_label_model() -> str:
    worker_ids, task_ids, times = read_task_times(dataset_path(repo_root()))
    return build_label_model(worker_ids, task_ids, times)


def build_lp_model(
    workers: list[int],
    tasks: list[str],
    task_times: dict[int, dict[str, str]],
) -> str:
    objective = [f"{task_times[worker][task]} x_{worker}_{task}" for worker in workers for task in tasks]
    lines = ["Minimize", f"    obj: {' + '.join(objective)}", "Subject To"]
    for task in tasks:
        lines.append(f"    assign_task_{task}: " + " + ".join(f"x_{worker}_{task}" for worker in workers) + " = 1")
    for worker in workers:
        lines.append(f"    worker_{worker}_limit: " + " + ".join(f"x_{worker}_{task}" for task in tasks) + " <= 1")
    lines.append(
        f"    total_assignments: "
        + " + ".join(f"x_{worker}_{task}" for worker in workers for task in tasks)
        + f" = {WORKERS_TO_SELECT}"
    )
    lines.extend(["Binaries", "    " + " ".join(f"x_{worker}_{task}" for worker in workers for task in tasks), "End"])
    return "\n".join(lines)


def generate_lp_model() -> str:
    worker_ids, task_ids, times = read_task_times(dataset_path(repo_root()))
    return build_lp_model(worker_ids, task_ids, times)

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
