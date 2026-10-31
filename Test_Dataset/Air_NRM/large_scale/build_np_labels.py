#!/usr/bin/env python3
"""Build and verify deterministic Air-NRM-NP-LS ground-truth labels.

The implementation uses gurobipy and never calls an LLM API.
Before building large-scale labels it reproduces all 21 shipped small-scale NP
objectives with both the approved tight sales/selection bound and the original
10,000,000 big-M formulation.

Run from any directory:
  python3 /absolute/path/to/build_np_labels.py
  python3 /absolute/path/to/build_np_labels.py --check
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

HERE = Path(__file__).resolve().parent
SMALL = HERE.parent / "small_scale"
QUERY = HERE / "query_NP_Flow.csv"
METADATA = HERE / "NP_query_metadata.json"
LABEL_DIR = HERE / "SBLP_NP_Flow_Label"
ANSWER = LABEL_DIR / "np_obj.csv"
PARAMETERS = LABEL_DIR / "NP_parameters.json"
SOLVER_REPORT = LABEL_DIR / "NP_solver_report.json"
VALIDATION_REPORT = HERE / "NP_LABEL_VALIDATION.md"
QUERY_FIELDS = ["n_variables", "Query", "Label-objective"]
LABEL_FIELDS = ["Obj_label", "X_sol_label", "Y_sol_label", "Information_label"]
TIME_LIMIT_S = 300.0
LEGACY_M = 10_000_000.0
TOL = 1e-6

CLAUSE = re.compile(
    r"\(OD\s*=\s*\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)\s*AND\s*"
    r"Departure\s+Time\s*=\s*'(\d{1,2}:\d{2})'\)")
CONSUME = re.compile(r"each\s+(Eco_\w+)\s+ticket\s+consumes\s+([\d.]+)\s+unit")
HAIRCUT = re.compile(r"capacity is ([\d.]+)% of its weekly economy capacity")
CARDINALITY = re.compile(r"(?:at most|optimal)\s+(\d+)\s+(?:selected\s+)?flights", re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    require(all(None not in row and all(value is not None for value in row.values())
                for row in rows), f"{path}: malformed CSV row")
    return reader.fieldnames, rows


def csv_text(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    return sha_bytes(path.read_bytes())


def normalize_time(value):
    h, m = map(int, value.split(":"))
    require(0 <= h < 24 and 0 <= m < 60, f"invalid time {value!r}")
    return f"{h:02d}:{m:02d}"


def parse_od(value):
    od = ast.literal_eval(value)
    require(isinstance(od, tuple) and len(od) == 2
            and all(isinstance(v, str) and v for v in od), f"invalid OD {value!r}")
    return od


def window_of(value):
    minutes = int(value[:2]) * 60 + int(value[3:])
    if 480 <= minutes < 720:
        return "(8am~12pm)"
    if 720 <= minutes < 1080:
        return "(12pm~6pm)"
    if 1080 <= minutes < 1320:
        return "(6pm~10pm)"
    return "(10pm~8am)"


def num(value, label, positive=False):
    result = float(value)
    require(math.isfinite(result) and (result > 0 if positive else result >= 0),
            f"invalid {'positive ' if positive else ''}{label}: {value}")
    return result


def safe(value):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(value)).strip("_")


class Data:
    def __init__(self, base, flat_capacity=None):
        v1_fields, v1_rows = read_csv(base / "v1.csv")
        self.products = []
        for field in v1_fields:
            if "*" in field and field.split("*", 1)[0] not in self.products:
                self.products.append(field.split("*", 1)[0])
        require(self.products, "v1.csv contains no product/window columns")
        self.short = {p: p.split("_")[-1][0] for p in self.products}
        require(len(set(self.short.values())) == len(self.products),
                f"product short-code collision: {self.products}")
        self.v1 = {parse_od(row["OD Pairs"]): row for row in v1_rows}
        self.v2 = {parse_od(row["OD Pairs"]): row
                   for row in read_csv(base / "v2.csv")[1]}
        self.demand = {parse_od(row["Oneway_OD"]): num(row["Avg Pax"], "demand", True)
                       for row in read_csv(base / "od_demand.csv")[1]}
        self.price = {}
        for row in read_csv(base / "flight.csv")[1]:
            key = (parse_od(row["Oneway_OD"]), normalize_time(row["Departure Time"]),
                   row["Oneway_Product"])
            require(key not in self.price, f"duplicate price row: {key}")
            self.price[key] = num(row["Avg Price"], "price", True)
        cap_path = base / "flight_capacity.csv"
        if cap_path.exists():
            self.capacity = {}
            for row in read_csv(cap_path)[1]:
                key = (parse_od(row["Oneway_OD"]), normalize_time(row["Departure Time"]))
                require(key not in self.capacity, f"duplicate capacity row: {key}")
                self.capacity[key] = num(row["Y Seats/Week"], "capacity")
            self.flat_capacity = None
        else:
            require(flat_capacity is not None, f"{base}: missing flight_capacity.csv")
            self.capacity, self.flat_capacity = {}, float(flat_capacity)


class Instance:
    def __init__(self, index, query, data, declared_variables=None):
        self.index, self.query, self.data = index, query, data
        self.query_sha256 = sha_bytes(query.encode())
        self.flights = [((o, d), normalize_time(t)) for o, d, t in CLAUSE.findall(query)]
        expected = len(re.findall(r"Departure\s+Time\s*=", query))
        require(self.flights and len(self.flights) == expected,
                f"q{index}: malformed flight clause")
        require(len(self.flights) == len(set(self.flights)), f"q{index}: duplicate flight")
        self.ods = sorted({od for od, _ in self.flights})
        self.airports = sorted({a for od in self.ods for a in od})
        z_hits = CARDINALITY.findall(query)
        require(z_hits and len(set(z_hits)) == 1, f"q{index}: ambiguous/missing selection limit")
        self.z = int(z_hits[0])
        consume = dict(CONSUME.findall(query))
        self.consumption = {}
        for product in data.products:
            key = next((k for k in consume
                        if k == product or product.startswith(k) or k.startswith(product)), None)
            require(key is not None, f"q{index}: missing consumption for {product}")
            self.consumption[data.short[product]] = num(consume[key], "consumption", True)
        cut = HAIRCUT.search(query)
        self.capacity_fraction = num(cut.group(1), "capacity fraction") / 100 if cut else 1.0
        self.cells, self.price, self.v1, self.v2 = [], {}, {}, {}
        self.v10, self.v20, self.demand, self.capacity = {}, {}, {}, {}
        for od in self.ods:
            require(od in data.v1 and od in data.v2 and od in data.demand,
                    f"q{index}: incomplete OD inputs {od}")
            self.v10[od] = num(data.v1[od]["no_purchase"], "v1 no_purchase", True)
            self.v20[od] = num(data.v2[od]["no_purchase"], "v2 no_purchase", True)
            self.demand[od] = data.demand[od]
        for od, departure in self.flights:
            raw_capacity = (data.flat_capacity if data.flat_capacity is not None
                            else data.capacity.get((od, departure)))
            require(raw_capacity is not None and raw_capacity > 0,
                    f"q{index}: missing/nonpositive capacity {(od, departure)}")
            self.capacity[(od, departure)] = raw_capacity * self.capacity_fraction
            for product in data.products:
                short = data.short[product]
                cell = (od, departure, short)
                price_key = (od, departure, product)
                require(price_key in data.price, f"q{index}: missing price {price_key}")
                field = product + "*" + window_of(departure)
                require(field in data.v1[od] and field in data.v2[od],
                        f"q{index}: missing attraction {od} {field}")
                self.cells.append(cell)
                self.price[cell] = data.price[price_key]
                self.v1[cell] = num(data.v1[od][field], "v1", True)
                self.v2[cell] = num(data.v2[od][field], "v2")
        self.n_variables = len(self.cells) + len(self.ods) + len(self.flights)
        if declared_variables is not None:
            require(int(declared_variables) == self.n_variables,
                    f"q{index}: declared {declared_variables}, built {self.n_variables} variables")

    def x_name(self, cell, lp=False):
        od, departure, short = cell
        if lp:
            return f"x_{safe(od[0])}_{safe(od[1])}_{departure.replace(':', '_')}_{short}"
        return f"x_{str(od).replace(' ', '_')}_{departure}_{short}"

    def x0_name(self, od, lp=False):
        if lp:
            return f"x0_{safe(od[0])}_{safe(od[1])}"
        return f"x_{str(od).replace(' ', '_')}_0"

    def y_name(self, flight, lp=False):
        od, departure = flight
        if lp:
            return f"y_{safe(od[0])}_{safe(od[1])}_{departure.replace(':', '_')}"
        return f"y_{str(od).replace(' ', '_')}_{departure}"

    def information(self):
        return str({"flights": [(str(od), departure) for od, departure in self.flights],
                    "n_ods": len(self.ods), "n_airports": len(self.airports),
                    "n_variables": self.n_variables, "Z": self.z,
                    "capacity_consum": self.consumption,
                    "capacity_fraction": self.capacity_fraction})

    def parameters(self):
        return {
            "query_sha256": self.query_sha256,
            "flights": [[*od, departure] for od, departure in self.flights],
            "products": self.data.products, "short_codes": self.data.short,
            "Z": self.z, "capacity_fraction": self.capacity_fraction,
            "capacity_consumption": self.consumption,
            "avg_price": {str(k): v for k, v in self.price.items()},
            "v1": {str(k): v for k, v in self.v1.items()},
            "v2": {str(k): v for k, v in self.v2.items()},
            "v1_no_purchase": {str(k): v for k, v in self.v10.items()},
            "v2_no_purchase": {str(k): v for k, v in self.v20.items()},
            "avg_pax": {str(k): v for k, v in self.demand.items()},
            "flight_capacity": {str(k): v for k, v in self.capacity.items()},
        }


def residuals(inst, solution):
    x, x0, y = solution["x"], solution["x0"], solution["y"]
    cap = max([sum(inst.consumption[s] * x[(od, t, s)]
                       for s in inst.data.short.values()) - inst.capacity[(od, t)]
               for od, t in inst.flights] + [0])
    scale = max([inst.v10[c[0]] * x[c] - inst.v1[c] * x0[c[0]]
                 for c in inst.cells] + [0])
    link = max([x[c] - inst.capacity[c[:2]] / inst.consumption[c[2]] * y[c[:2]]
                for c in inst.cells] + [0])
    balance = max([abs(sum(inst.v2[c] * x[c] for c in inst.cells if c[0] == od)
                           + inst.v20[od] * x0[od] - inst.demand[od])
                   for od in inst.ods] + [0])
    flow = max([abs(sum(y[f] for f in inst.flights if f[0][0] == airport)
                        - sum(y[f] for f in inst.flights if f[0][1] == airport))
                for airport in inst.airports] + [0])
    cardinality = max(0, sum(y.values()) - inst.z)
    integrality = max([abs(value - round(value)) for value in y.values()] + [0])
    negativity = max([max(0, -value) for value in [*x.values(), *x0.values(), *y.values()]] + [0])
    return {"capacity_excess": cap, "scale_excess": scale, "selection_excess": link,
            "balance_abs": balance, "flow_abs": flow, "cardinality_excess": cardinality,
            "binary_integrality": integrality, "negativity": negativity}


def make_model(inst, legacy_m=False):
    """Build the exact NP model in gurobipy; no data are read inside the model."""
    model = gp.Model(f"Air_NRM_NP_LS_{inst.index}")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = TIME_LIMIT_S
    model.Params.MIPGap = 0.0
    x = {cell: model.addVar(lb=0, name=inst.x_name(cell, True)) for cell in inst.cells}
    x0 = {od: model.addVar(lb=0, name=inst.x0_name(od, True)) for od in inst.ods}
    y = {flight: model.addVar(vtype=GRB.BINARY, name=inst.y_name(flight, True))
         for flight in inst.flights}
    model.setObjective(gp.quicksum(inst.price[cell] * x[cell] for cell in inst.cells),
                       GRB.MAXIMIZE)
    for od, departure in inst.flights:
        model.addConstr(gp.quicksum(
            inst.consumption[s] * x[(od, departure, s)]
            for s in inst.data.short.values()) <= inst.capacity[(od, departure)],
            name=f"capacity_{safe(od)}_{departure.replace(':', '_')}")
    for cell in inst.cells:
        od = cell[0]
        model.addConstr(inst.v10[od] * x[cell] <= inst.v1[cell] * x0[od],
                        name=f"scale_{safe(cell)}")
    for cell in inst.cells:
        flight = cell[:2]
        upper = LEGACY_M if legacy_m else inst.capacity[flight] / inst.consumption[cell[2]]
        model.addConstr(x[cell] <= upper * y[flight], name=f"selection_{safe(cell)}")
    for od in inst.ods:
        model.addConstr(
            gp.quicksum(inst.v2[cell] * x[cell] for cell in inst.cells if cell[0] == od)
            + inst.v20[od] * x0[od] == inst.demand[od], name=f"balance_{safe(od)}")
    for airport in inst.airports:
        outbound = gp.quicksum(y[f] for f in inst.flights if f[0][0] == airport)
        inbound = gp.quicksum(y[f] for f in inst.flights if f[0][1] == airport)
        model.addConstr(outbound == inbound, name=f"flow_{safe(airport)}")
    model.addConstr(gp.quicksum(y.values()) <= inst.z, name="cardinality")
    model.update()
    return model, x, x0, y


def solve(inst, legacy_m=False, lp_path=None):
    model, x_vars, x0_vars, y_vars = make_model(inst, legacy_m=legacy_m)
    expected_constraints = (len(inst.flights) + 2 * len(inst.cells) + len(inst.ods)
                            + len(inst.airports) + 1)
    require(model.NumVars == inst.n_variables, f"q{inst.index}: Gurobi variable count differs")
    require(model.NumConstrs == expected_constraints,
            f"q{inst.index}: Gurobi constraint count differs")
    if lp_path is not None:
        lp_path.parent.mkdir(parents=True, exist_ok=True)
        model.write(str(lp_path))
    started = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - started
    require(model.Status == GRB.OPTIMAL,
            f"q{inst.index}: Gurobi did not certify optimality (status {model.Status})")
    objective, best_bound, gap = float(model.ObjVal), float(model.ObjBound), float(model.MIPGap)
    scale = max(1.0, abs(objective))
    require(abs(best_bound - objective) <= max(1e-5, 1e-8 * scale),
            f"q{inst.index}: incumbent and optimal bound differ")
    x = {cell: float(var.X) for cell, var in x_vars.items()}
    x0 = {od: float(var.X) for od, var in x0_vars.items()}
    y = {flight: float(var.X) for flight, var in y_vars.items()}
    recomputed = sum(inst.price[cell] * x[cell] for cell in inst.cells)
    require(abs(recomputed - objective) <= max(1e-5, 1e-8 * scale),
            f"q{inst.index}: recomputed objective differs")
    checks = residuals(inst, {"x": x, "x0": x0, "y": y})
    require(checks["balance_abs"] <= 1e-7 * max(inst.demand.values()),
            f"q{inst.index}: balance residual {checks['balance_abs']}")
    require(max(v for k, v in checks.items() if k != "balance_abs") <= 1e-5,
            f"q{inst.index}: feasibility residuals {checks}")

    fixed = model.fixed()
    fixed.Params.OutputFlag = 0
    fixed.Params.TimeLimit = TIME_LIMIT_S
    fixed.optimize()
    require(fixed.Status == GRB.OPTIMAL
            and abs(float(fixed.ObjVal) - objective) <= max(1e-5, 1e-8 * scale),
            f"q{inst.index}: fixed-selection LP cross-check failed")
    fixed_objective = float(fixed.ObjVal)
    nodes, runtime = int(model.NodeCount), float(model.Runtime)
    message = "OPTIMAL"
    selected = [f for f, value in y.items() if value > 0.5]
    fixed.dispose()
    model.dispose()
    return {"objective": objective, "best_bound": best_bound, "mip_gap": gap,
            "mip_node_count": nodes, "solve_seconds": elapsed,
            "solver_runtime_seconds": runtime, "message": message,
            "x": x, "x0": x0, "y": y, "selected_flights": selected,
            "residuals": checks, "fixed_selection_lp_objective": fixed_objective,
            "legacy_m": legacy_m, "constraints": expected_constraints}


def verify_written_lp(inst, lp_path, expected_objective):
    text = lp_path.read_text()
    require("Maximize" in text and "Subject To" in text and "Binaries" in text,
            f"q{inst.index}: incomplete Gurobi LP file")
    require("10000000" not in text, f"q{inst.index}: legacy big-M leaked into LP")
    reloaded = gp.read(str(lp_path))
    reloaded.Params.OutputFlag = 0
    reloaded.Params.TimeLimit = TIME_LIMIT_S
    reloaded.Params.MIPGap = 0.0
    require(reloaded.NumVars == inst.n_variables, f"q{inst.index}: reloaded LP variable count")
    expected_constraints = (len(inst.flights) + 2 * len(inst.cells) + len(inst.ods)
                            + len(inst.airports) + 1)
    require(reloaded.NumConstrs == expected_constraints,
            f"q{inst.index}: reloaded LP constraint count")
    reloaded.optimize()
    require(reloaded.Status == GRB.OPTIMAL, f"q{inst.index}: reloaded LP not optimal")
    require(abs(float(reloaded.ObjVal) - expected_objective)
            <= 1e-7 * max(1, abs(expected_objective)),
            f"q{inst.index}: reloaded LP objective differs")
    reloaded.dispose()


def small_scale_validation():
    query_fields, query_rows = read_csv(SMALL / "query_NP_Flow.csv")
    _, reference = read_csv(SMALL / "SBLP_NP_Flow_Label" / "np_obj.csv")
    require(len(query_rows) == len(reference) == 21, "small-scale NP row count mismatch")
    data = Data(SMALL, flat_capacity=187.0)
    rows, worst, legacy_worst = [], 0.0, 0.0
    for i, (query_row, label) in enumerate(zip(query_rows, reference), 1):
        declared = query_row.get("Decision Variables")
        inst = Instance(i, query_row["Query"], data, declared)
        tight = solve(inst)
        legacy = solve(inst, legacy_m=True)
        gold = float(label["Obj_label"])
        rel = abs(tight["objective"] - gold) / max(1, abs(gold))
        legacy_rel = abs(tight["objective"] - legacy["objective"]) / max(1, abs(gold))
        worst, legacy_worst = max(worst, rel), max(legacy_worst, legacy_rel)
        require(rel < TOL, f"small q{i}: objective differs from shipped label by {rel}")
        require(legacy_rel < TOL, f"small q{i}: tight/legacy M differ by {legacy_rel}")
        rows.append({"instance": i, "variables": inst.n_variables,
                     "shipped_objective": gold, "rebuilt_objective": tight["objective"],
                     "relative_error": rel, "tight_vs_legacy_relative_error": legacy_rel})
    return {"instances": rows, "count": len(rows), "worst_relative_error": worst,
            "worst_tight_vs_legacy_relative_error": legacy_worst,
            "status": "pass", "query_columns": query_fields}


def query_metadata_check(instances):
    metadata = json.loads(METADATA.read_text())
    require(metadata["n_queries"] == len(instances), "NP metadata row count mismatch")
    for inst, saved in zip(instances, metadata["instances"]):
        require(inst.index - 1 == saved["instance_id"], f"q{inst.index}: metadata id mismatch")
        require(inst.query_sha256 == saved["np_query_sha256"], f"q{inst.index}: query hash mismatch")
        require(inst.n_variables == saved["n_variables"] and inst.z == saved["Z"],
                f"q{inst.index}: metadata variables/Z mismatch")
        require([[*od, t] for od, t in inst.flights] == saved["flights"],
                f"q{inst.index}: metadata candidate set mismatch")


def report_markdown(report):
    small = report["small_scale_validation"]
    lines = ["# Task 3: Air-NRM Large Scale NP 标签验证报告", "",
             "标准标签由确定性 SBLP/MILP 程序和本地 gurobipy/Gurobi 生成，未调用模型 API。",
             "Gurobi 对每题返回最优状态；保存的 incumbent 与最优界一致，MIP gap 为零。", "",
             f"Small scale 回归：{small['count']}/21 通过；相对原标签的最坏误差 "
             f"`{small['worst_relative_error']:.3e}`；紧联动上界与原 `M=10,000,000` 的最坏差异 "
             f"`{small['worst_tight_vs_legacy_relative_error']:.3e}`。", "",
             "当前最大题为 621 个变量、1,106 条线性约束。当前 Gurobi restricted license 的上限是",
             "2,000 个变量和 2,000 条线性约束；按第 15 题的 OD/机场构成，约束数先触顶，可扩展到约",
             "214 个候选航班。Gurobi 学术许可完整且无模型规模限制。官方说明：",
             "https://support.gurobi.com/hc/en-us/articles/29682074018833-What-does-Restricted-license-for-non-production-use-only-mean",
             "以及 https://www.gurobi.com/product/pricing-and-licensing 。", "",
             "| 题号 | 航班 | OD | 机场 | 变量 | 约束 | Z | 选中 | 最优收入 | 最优界 | gap | 时间(s) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in report["large_scale"]:
        lines.append("| " + " | ".join([
            str(row["instance"]), str(row["flights"]), str(row["ods"]),
            str(row["airports"]), str(row["variables"]), str(row["constraints"]),
            str(row["Z"]), str(row["selected"]), f"{row['objective']:.6f}",
            f"{row['best_bound']:.6f}", f"{row['mip_gap']:.1e}",
            f"{row['solve_seconds']:.3f}"]) + " |")
    lines += ["", "每题额外完成：变量计数、Task 2 query/候选集哈希、容量/scale/balance/选班联动/流守恒/",
              "cardinality/非负/二元性残差检查，以及固定选班后的连续 LP 再优化核对。", "",
              "输出：`SBLP_NP_Flow_Label/1.lp` 到 `15.lp`、`np_obj.csv`、`NP_parameters.json`、",
              "`NP_solver_report.json`，并将同一最优收入写入 `query_NP_Flow.csv` 的 `Label-objective`。", "",
              "## 本地复核", "", "无需 API：", "", "```bash",
              "cd /Users/yikeshuilingnencong/Desktop/lean-llm-opt-main",
              "/opt/anaconda3/bin/python Test_Dataset/Air_NRM/large_scale/build_np_labels.py --check", "```", ""]
    return "\n".join(lines)


def validate_saved_label(inst, row, expected_objective):
    """Validate a saved solution without requiring the same alternate optimum."""
    require(abs(float(row["Obj_label"]) - expected_objective)
            <= 1e-7 * max(1, abs(expected_objective)),
            f"q{inst.index}: saved objective differs")
    saved_x, saved_y = json.loads(row["X_sol_label"]), json.loads(row["Y_sol_label"])
    x_names = {inst.x_name(c): c for c in inst.cells}
    x0_names = {inst.x0_name(od): od for od in inst.ods}
    y_names = {inst.y_name(f): f for f in inst.flights}
    require(set(saved_x) <= set(x_names) | set(x0_names),
            f"q{inst.index}: saved X solution has unknown variables")
    require(set(saved_y) <= set(y_names), f"q{inst.index}: saved Y solution has unknown variables")
    x = {c: float(saved_x.get(name, 0)) for name, c in x_names.items()}
    x0 = {od: float(saved_x.get(name, 0)) for name, od in x0_names.items()}
    y = {f: float(saved_y.get(name, 0)) for name, f in y_names.items()}
    checks = residuals(inst, {"x": x, "x0": x0, "y": y})
    require(checks["balance_abs"] <= 1e-7 * max(inst.demand.values()),
            f"q{inst.index}: saved solution balance residual")
    require(max(v for k, v in checks.items() if k != "balance_abs") <= 1e-5,
            f"q{inst.index}: saved solution infeasible: {checks}")
    revenue = sum(inst.price[c] * x[c] for c in inst.cells)
    require(abs(revenue - expected_objective) <= 1e-7 * max(1, abs(expected_objective)),
            f"q{inst.index}: saved allocation revenue differs")


def build(check=False):
    small_report = small_scale_validation()
    fields, query_rows = read_csv(QUERY)
    require(fields == QUERY_FIELDS and len(query_rows) == 15,
            f"large NP query schema/count mismatch: {fields}, {len(query_rows)}")
    data = Data(HERE)
    instances = [Instance(i, row["Query"], data, row["n_variables"])
                 for i, row in enumerate(query_rows, 1)]
    query_metadata_check(instances)
    output_rows, parameter_rows, large_report, lp_outputs = [], [], [], {}
    lp_temp = tempfile.TemporaryDirectory(prefix="air-nrm-np-gurobi-lp-", dir=HERE)
    for inst in instances:
        temp_lp = Path(lp_temp.name) / f"{inst.index}.lp"
        solved = solve(inst, lp_path=temp_lp)
        verify_written_lp(inst, temp_lp, solved["objective"])
        lp_bytes = temp_lp.read_bytes()
        x_solution = {inst.x_name(c): value for c, value in solved["x"].items() if value > 1e-9}
        x_solution.update({inst.x0_name(od): value for od, value in solved["x0"].items()
                           if value > 1e-9})
        y_solution = {inst.y_name(f): value for f, value in solved["y"].items() if value > 0.5}
        output_rows.append({"Obj_label": f"{solved['objective']:.15g}",
                            "X_sol_label": json.dumps(x_solution, sort_keys=True),
                            "Y_sol_label": json.dumps(y_solution, sort_keys=True),
                            "Information_label": inst.information()})
        parameter_rows.append({"instance": inst.index, **inst.parameters()})
        constraints = (len(inst.flights) + 2 * len(inst.cells) + len(inst.ods)
                       + len(inst.airports) + 1)
        large_report.append({
            "instance": inst.index, "query_sha256": inst.query_sha256,
            "flights": len(inst.flights), "ods": len(inst.ods),
            "airports": len(inst.airports), "variables": inst.n_variables,
            "continuous_variables": len(inst.cells) + len(inst.ods),
            "binary_variables": len(inst.flights), "constraints": constraints,
            "Z": inst.z, "selected": len(solved["selected_flights"]),
            "selected_flights": [[*f[0], f[1]] for f in solved["selected_flights"]],
            "objective": solved["objective"], "best_bound": solved["best_bound"],
            "mip_gap": solved["mip_gap"], "mip_node_count": solved["mip_node_count"],
            "solve_seconds": solved["solve_seconds"], "solver_message": solved["message"],
            "fixed_selection_lp_objective": solved["fixed_selection_lp_objective"],
            "residuals": solved["residuals"], "lp_sha256": sha_bytes(lp_bytes),
        })
        lp_outputs[inst.index] = lp_bytes
        print(f"q{inst.index:02d}: vars={inst.n_variables:3d}  Z={inst.z:2d}  "
              f"selected={len(solved['selected_flights']):2d}  "
              f"objective={solved['objective']:,.6f}  gap={solved['mip_gap']:.1e}  "
              f"{solved['solve_seconds']:.2f}s")
    report = {
        "schema_version": 1, "dataset": "Air-NRM-NP-LS", "solver": "gurobipy/Gurobi",
        "gurobi_version": list(gp.gurobi.version()), "time_limit_s": TIME_LIMIT_S,
        "formulation": {"selection_link": "x[l,k,j] <= capacity[l,k]/consumption[j] * y[l,k]",
                        "cardinality": "sum(y) <= Z",
                        "flow": "selected outbound count = selected inbound count per airport"},
        "license": {"observed_type": "Restricted license - for non-production use only",
                    "max_variables": 2000, "max_linear_constraints": 2000,
                    "largest_instance_variables": max(i.n_variables for i in instances),
                    "largest_instance_constraints": max(
                        len(i.flights) + 2 * len(i.cells) + len(i.ods) + len(i.airports) + 1
                        for i in instances),
                    "q15_fixed_od_airport_estimated_max_flights": 214,
                    "academic_license": "full-featured, no model-size limit"},
        "source_sha256": {name: sha_file(HERE / name) for name in
                          ["NP_query_metadata.json", "flight.csv", "flight_capacity.csv",
                           "od_demand.csv", "v1.csv", "v2.csv"]},
        "small_scale_validation": small_report, "large_scale": large_report,
    }
    report["query_semantic_sha256"] = sha_bytes(csv_text(
        ["n_variables", "Query"],
        [{"n_variables": row["n_variables"], "Query": row["Query"]}
         for row in query_rows]).encode())
    answer_text = csv_text(LABEL_FIELDS, output_rows)
    parameter_text = json.dumps({"schema_version": 1, "dataset": "Air-NRM-NP-LS",
                                 "instances": parameter_rows}, indent=2) + "\n"

    labeled_queries = []
    for source, solved in zip(query_rows, large_report):
        labeled_queries.append({**source, "Label-objective": f"{solved['objective']:.15g}"})
    query_text = csv_text(QUERY_FIELDS, labeled_queries)
    # The saved source hash intentionally refers to the pre-label query file.
    report_text = json.dumps(report, indent=2) + "\n"
    validation_text = report_markdown(report)
    if check:
        saved_fields, saved_answers = read_csv(ANSWER)
        require(saved_fields == LABEL_FIELDS and len(saved_answers) == len(instances),
                "np_obj.csv schema/count differs")
        for inst, row, solved in zip(instances, saved_answers, large_report):
            validate_saved_label(inst, row, solved["objective"])
        require(PARAMETERS.read_text() == parameter_text, "NP_parameters.json differs from rebuilt parameters")
        require(QUERY.read_text() == query_text, "query_NP_Flow.csv objectives differ from rebuilt labels")
        saved_report = json.loads(SOLVER_REPORT.read_text())
        # Runtime, node count and equivalent optimal selections can vary.
        # Compare stable facts and solve every Gurobi-written LP again.
        require(saved_report["dataset"] == report["dataset"]
                and saved_report["source_sha256"] == report["source_sha256"]
                and saved_report["query_semantic_sha256"] == report["query_semantic_sha256"],
                "solver report source identity differs")
        for saved, fresh in zip(saved_report["large_scale"], large_report):
            for key in ["instance", "query_sha256", "flights", "ods", "airports",
                        "variables", "binary_variables", "constraints", "Z"]:
                require(saved[key] == fresh[key], f"q{fresh['instance']}: saved {key} differs")
            require(abs(saved["objective"] - fresh["objective"]) <= 1e-6 * max(1, abs(fresh["objective"])),
                    f"q{fresh['instance']}: rebuilt objective differs")
            lp_path = LABEL_DIR / f"{fresh['instance']}.lp"
            require(sha_file(lp_path) == saved["lp_sha256"], f"q{fresh['instance']}: LP file hash differs")
            verify_written_lp(instances[fresh["instance"] - 1], lp_path, fresh["objective"])
        lp_temp.cleanup()
        print("PASS: saved labels, query objectives, parameters and 15 LP models verified.")
        return

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    # Write complete results only after all 15 instances have optimal solutions.
    with tempfile.TemporaryDirectory(prefix="air-nrm-np-labels-", dir=HERE) as tmp_name:
        tmp = Path(tmp_name)
        (tmp / "np_obj.csv").write_text(answer_text)
        (tmp / "NP_parameters.json").write_text(parameter_text)
        (tmp / "NP_solver_report.json").write_text(report_text)
        for i, lp in lp_outputs.items():
            (tmp / f"{i}.lp").write_bytes(lp)
        for path in tmp.iterdir():
            path.replace(LABEL_DIR / path.name)
    QUERY.write_text(query_text)
    VALIDATION_REPORT.write_text(validation_text)
    lp_temp.cleanup()
    print("PASS: wrote 15 optimal NP labels, LP models, parameters and validation report.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="solve again and compare saved outputs")
    args = parser.parse_args()
    build(check=args.check)


if __name__ == "__main__":
    main()
