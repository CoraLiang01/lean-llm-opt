#!/usr/bin/env python3
"""Build/check 15 NP queries aligned with CA; never call an LLM API.

Uses scipy.optimize.milp (SciPy >= 1.9) only to certify maximum circulation
cardinality B. This is not the revenue-label solver. Label-objective starts
blank and is preserved on a subsequent build only if all query bodies match.

Run from any directory: python3 /absolute/path/to/build_np_queries.py [--check]
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
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ["n_variables", "Query", "Label-objective"]
INPUTS = ["query_CA.csv", "flight.csv", "flight_capacity.csv",
          "od_demand.csv", "v1.csv", "v2.csv"]
PRODUCTS = ["Eco_flexi", "Eco_standard", "Eco_value", "Eco_lite"]
CLAUSE = re.compile(
    r"\(OD\s*=\s*\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)\s*AND\s*"
    r"Departure\s+Time\s*=\s*'(\d{1,2}:\d{2})'\)")
CONSUME = re.compile(r"each\s+(Eco_\w+)\s+ticket\s+consumes\s+([\d.]+)\s+unit")
HAIRCUT = re.compile(r"capacity is ([\d.]+)% of its weekly economy capacity")
Flight = tuple[str, str, str]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    require(all(None not in r and all(v is not None for v in r.values())
                for r in rows), f"{path.name}: malformed CSV row")
    return reader.fieldnames, rows


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalize_time(value):
    h, m = map(int, value.split(":"))
    require(0 <= h < 24 and 0 <= m < 60, f"invalid time: {value}")
    return f"{h:02d}:{m:02d}"


def od_pair(value):
    od = ast.literal_eval(value)
    require(isinstance(od, tuple) and len(od) == 2
            and all(isinstance(x, str) and x for x in od)
            and od[0] != od[1], f"invalid OD: {value}")
    return od


def flight_key(row):
    return (*od_pair(row["Oneway_OD"]), normalize_time(row["Departure Time"]))


def parse_flights(query):
    flights = [(o, d, normalize_time(t)) for o, d, t in CLAUSE.findall(query)]
    expected = len(re.findall(r"Departure\s+Time\s*=", query))
    require(flights and len(flights) == expected, "missing or malformed flight clause")
    require(len(flights) == len(set(flights)), "duplicate flight in query")
    return flights


def time_window(t):
    minutes = int(t[:2]) * 60 + int(t[3:])
    if 480 <= minutes < 720:
        return "(8am~12pm)"
    if 720 <= minutes < 1080:
        return "(12pm~6pm)"
    if 1080 <= minutes < 1320:
        return "(6pm~10pm)"
    return "(10pm~8am)"


def number(value, label, positive=False):
    result = float(value)
    require(math.isfinite(result) and (result > 0 if positive else result >= 0),
            f"invalid {'positive ' if positive else ''}number for {label}: {value}")
    return result


class Data:
    def __init__(self, base):
        self.prices = defaultdict(dict)
        for r in read_csv(base / "flight.csv")[1]:
            key, product = flight_key(r), r["Oneway_Product"]
            require(product not in self.prices[key], f"duplicate price: {key} {product}")
            self.prices[key][product] = number(r["Avg Price"], "price", positive=True)
        self.capacity = {}
        for r in read_csv(base / "flight_capacity.csv")[1]:
            key = flight_key(r)
            require(key not in self.capacity, f"duplicate capacity: {key}")
            self.capacity[key] = number(r["Y Seats/Week"], "weekly capacity")
        self.demand, self.v1, self.v2 = {}, {}, {}
        for name, target, field in [("od_demand.csv", self.demand, "Oneway_OD"),
                                    ("v1.csv", self.v1, "OD Pairs"),
                                    ("v2.csv", self.v2, "OD Pairs")]:
            for r in read_csv(base / name)[1]:
                od = od_pair(r[field])
                require(od not in target, f"{name}: duplicate OD {od}")
                target[od] = r
        self.pool = defaultdict(list)
        for f in sorted(self.prices):
            if self.capacity.get(f, 0) > 0:
                self.validate(f)
                self.pool[f[:2]].append(f)

    def validate(self, f):
        require(set(self.prices.get(f, {})) == set(PRODUCTS),
                f"{f}: expected all four named products")
        require(self.capacity.get(f, 0) > 0, f"{f}: no positive matched capacity")
        od = f[:2]
        require(od in self.demand and od in self.v1 and od in self.v2,
                f"{f}: missing demand/attraction OD")
        number(self.demand[od]["Avg Pax"], "demand", positive=True)
        for name, table in [("v1", self.v1), ("v2", self.v2)]:
            number(table[od]["no_purchase"], f"{name} no_purchase", positive=True)
            for p in PRODUCTS:
                value = number(table[od][p + "*" + time_window(f[2])],
                               f"{f} {name} {p}", positive=name == "v1")
                if name == "v2":
                    require(value <= 1, f"{f} {p}: v2 product ratio exceeds 1")


def flow_deltas(flights):
    delta = Counter()
    for o, d, _ in flights:
        delta[o] += 1
        delta[d] -= 1
    return dict(sorted(delta.items()))


def maximum_circulation(flights):
    """Certify max sum(y), incidence*y=0, y binary; no revenue parameters."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    airports = sorted({a for f in flights for a in f[:2]})
    index = {a: i for i, a in enumerate(airports)}
    incidence = np.zeros((len(airports), len(flights)))
    for j, (o, d, _) in enumerate(flights):
        incidence[index[o], j] = 1
        incidence[index[d], j] = -1
    result = milp(
        -np.ones(len(flights)), integrality=np.ones(len(flights)),
        bounds=Bounds(0, 1), constraints=LinearConstraint(incidence, 0, 0),
        options={"time_limit": 60, "mip_rel_gap": 0.0},
    )
    require(result.status == 0 and result.x is not None,
            f"maximum circulation not certified optimal: {result.message}")
    require(np.max(np.abs(result.x - np.rint(result.x))) <= 1e-7,
            "maximum circulation solution is not integral")
    selected = [f for f, y in zip(flights, result.x) if y > 0.5]
    require(not any(flow_deltas(selected).values()), "unbalanced circulation witness")
    require(abs(result.fun + len(selected)) <= 1e-7, "cardinality objective mismatch")
    require(abs(result.mip_dual_bound + len(selected)) <= 1e-7,
            "maximum circulation has no matching optimal bound")
    return len(selected), selected


def make_query(flights, z, capacity_tail):
    clauses = ", ".join(
        f"(OD = {str((o, d))} AND Departure Time='{t}')" for o, d, t in flights)
    return (
        "Based on all flight ticket choices in 'flight.csv' and 'od_demand.csv' "
        "with attraction values in v1 and shadow attraction value ratios in v2, "
        "develop a Sales-Based Linear Programming (SBLP) model for joint fare "
        "type capacity allocation and flight scheduling, while enforcing flow "
        "conservation constraints at each airport for a long-term planning model. "
        f"The goal is to recommend at most {z} flights that maximize the total "
        f"revenue of flight ticket sales, specifically among flights {clauses}. "
        "The SBLP should include decision variables, an objective function, "
        "capacity constraints, balance constraints, scale constraints, "
        "nonnegative constraints, binary flight selection variables, constraints "
        "ensuring that ticket sales are zero for any unselected flight, a "
        f"cardinality constraint allowing at most {z} selected flights in total, "
        "and flow conservation constraints requiring the number of selected "
        "inbound flights to equal the number of selected outbound flights at "
        "each airport. Each distinct (Oneway_OD, Departure Time) option counts "
        "as one flight for selection and flow conservation. " + capacity_tail
    )


def build(base):
    header, ca = read_csv(base / "query_CA.csv")
    require(header == FIELDS, f"unexpected CA columns: {header}")
    require(len(ca) == 15, f"expected 15 CA queries, got {len(ca)}")
    inputs = {name: sha((base / name).read_bytes()) for name in INPUTS}
    data = Data(base)
    rows, instances = [], []
    for qi, source in enumerate(ca, 1):
        original = parse_flights(source["Query"])
        ods = {f[:2] for f in original}
        require(int(source["n_variables"]) == 4 * len(original) + len(ods),
                f"q{qi}: source CA variable count mismatch")
        for f in original:
            data.validate(f)
        missing = sorted({(d, o) for o, d in ods} - ods)
        added = []
        for od in missing:
            require(bool(data.pool[od]), f"q{qi}: no eligible reverse flight for {od}")
            added.append(data.pool[od][0])
        flights = original + added
        require(len(flights) == len(set(flights)), f"q{qi}: duplicate augmented flight")
        final_ods = {f[:2] for f in flights}
        require(all((d, o) in final_ods for o, d in final_ods), f"q{qi}: missing reverse OD")
        b, witness = maximum_circulation(flights)
        z = b // 2
        # Each augmented OD has a reverse, giving an explicit 2-edge witness.
        first = min(flights)
        reverse = min(f for f in flights if f[:2] == (first[1], first[0]))
        nonempty = [first, reverse]
        require(z >= len(nonempty) and not any(flow_deltas(nonempty).values()),
                f"q{qi}: no nonempty witnessed circulation under Z={z}")
        tail_match = re.search(r"Each flight's capacity is.*", source["Query"], re.S)
        require(tail_match is not None, f"q{qi}: missing capacity specification")
        tail = tail_match.group().strip()
        consumption_matches = CONSUME.findall(tail)
        consumption = {k: number(v, k, positive=True) for k, v in consumption_matches}
        require(len(consumption_matches) == 4 and set(consumption) == set(PRODUCTS),
                f"q{qi}: consumption coefficients incomplete or repeated")
        haircut = HAIRCUT.search(tail)
        require(haircut is not None and float(haircut[1]) == 28, f"q{qi}: expected 28% capacity")
        text = make_query(flights, z, tail)
        require(parse_flights(text) == flights, f"q{qi}: rendered candidate list differs")
        n_variables = 5 * len(flights) + len(final_ods)
        rows.append({"n_variables": str(n_variables), "Query": text, "Label-objective": ""})
        instances.append({
            "instance_id": qi - 1, "source_ca_row_1based": qi,
            "source_query_sha256": sha(source["Query"].encode()),
            "np_query_sha256": sha(text.encode()),
            "source_ca_n_variables": int(source["n_variables"]),
            "n_variables": n_variables, "n_sales_variables": 4 * len(flights),
            "n_no_purchase_variables": len(final_ods), "n_binary_variables": len(flights),
            "n_airports": len({a for f in flights for a in f[:2]}),
            "n_ods": len(final_ods), "n_original_flights": len(original),
            "n_added_flights": len(added), "n_flights": len(flights),
            "original_flights": original, "added_flights": added, "flights": flights,
            "capacity_fraction": 0.28, "capacity_consumption": consumption,
            "B": b, "Z": z, "B_optimal_bound": b,
            "maximum_circulation_witness": witness,
            "nonempty_under_Z_witness": nonempty,
            "checks": {"original_flights_preserved": True,
                       "all_five_data_files_joined": True,
                       "all_candidates_on_a_directed_cycle": True,
                       "B_proven_optimal": True, "nonempty_under_Z": True,
                       "query_roundtrip": True},
        })
    require(inputs == {name: sha((base / name).read_bytes()) for name in INPUTS},
            "source files changed during generation")
    metadata = {
        "schema_version": 1, "dataset": "Air-NRM-NP-LS", "n_queries": 15,
        "csv_columns": FIELDS, "label_policy": "blank until Task 3; CA labels are never copied",
        "input_sha256": inputs,
        "flight_representation": ["origin", "destination", "departure_time"],
        "policy": {"augmentation": "one earliest positive-capacity flight per missing reverse OD",
                   "ordering": "original CA order, then added flights sorted by OD and time",
                   "B": "max sum(y) subject to airport incidence*y=0 and y binary",
                   "Z": "floor(B/2)", "solver": "scipy.optimize.milp / HiGHS, zero relative gap",
                   "revenue_optimization_performed": False},
        "instances": instances,
    }
    return rows, metadata


def report_text(meta):
    lines = ["# Task 2: Large Scale NP query 验证报告", "",
             "15 条 NP query 与 CA 原行顺序一一对应；CSV 列名与顺序完全一致：",
             "`n_variables,Query,Label-objective`。", "",
             "`Label-objective` 初始留空，等待 Task 3 的 NP 收入标签；未复制 CA 目标值。",
             "所有原候选航班和容量条款均保留。缺少的反向 OD 按最早可用出发时间补入真实航班。",
             "B 由本地整数网络流求解并确认最优，Z=floor(B/2)。每题另保存一个满足 Z 的非空闭环见证。",
             "未调用 API，未求解 NP 收入目标。", "",
             "| CA/NP 题号 | 原航班 | 补入 | 最终航班 | OD | 机场 | 变量 | B | Z | 检查 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in meta["instances"]:
        values = [r[k] for k in ["source_ca_row_1based", "n_original_flights", "n_added_flights",
                                  "n_flights", "n_ods", "n_airports", "n_variables", "B", "Z"]]
        lines.append("| " + " | ".join(map(str, values)) + " | 通过 |")
    lines += ["", "每题检查：原 CA 航班保留、无重复、四种价格产品齐全、容量为正、需求及 v1/v2 可连接、",
              "返向 OD 完整、最大闭环选班数最优性、Z 下非空闭环、自然语言候选集回读及变量计数。", "",
              "补入明细、原/新 query 哈希、六个源文件 SHA-256、最大流守恒选班见证均在 `NP_query_metadata.json`。",
              "此处的闭环检查针对航班选择；完整 SBLP 的约束及收入验证在 Task 3 进行。", "",
              "## 本地命令", "", "从仓库根目录执行（需要 SciPy >= 1.9，不需要 API）：", "", "```bash",
              "python3 Test_Dataset/Air_NRM/large_scale/build_np_queries.py --check", "```", "",
              "重新生成时去掉 `--check`。已有非空收入标签只在全部 query 和变量数一致时保留；",
              "query 已改变时拒绝覆盖已有标签。检查模式不写文件，并核验已保存的元数据和见证。", ""]
    return "\n".join(lines)


def existing_labels(path, rows):
    if not path.exists():
        return
    header, existing = read_csv(path)
    require(header == FIELDS, "existing NP CSV header differs from CA")
    labeled = any(r["Label-objective"].strip() for r in existing)
    matches = (len(existing) == len(rows) and all(
        old["Query"] == new["Query"] and old["n_variables"] == new["n_variables"]
        for old, new in zip(existing, rows)))
    require(not labeled or matches, "NP queries changed: refusing to overwrite existing revenue labels")
    if matches:
        for old, new in zip(existing, rows):
            if old["Label-objective"].strip():
                number(old["Label-objective"], "existing NP label")
            new["Label-objective"] = old["Label-objective"]


def check_metadata(actual, expected):
    """Compare reproducible facts; verify alternate optimal solver witnesses too."""
    require(len(actual.get("instances", [])) == 15, "metadata instance count differs")
    # HiGHS versions may select different equally optimal flight sets.
    for saved, fresh in zip(actual["instances"], expected["instances"]):
        witness = [tuple(f) for f in saved["maximum_circulation_witness"]]
        pool = {tuple(f) for f in fresh["flights"]}
        require(len(set(witness)) == len(witness) == fresh["B"]
                and set(witness) <= pool and not any(flow_deltas(witness).values()),
                "saved maximum circulation witness is invalid")
        saved["maximum_circulation_witness"] = fresh["maximum_circulation_witness"]
    # JSON roundtrip normalizes Python tuples to CSV-independent arrays.
    require(json.loads(json.dumps(actual)) == json.loads(json.dumps(expected)),
            "metadata is stale or inconsistent with the CA inputs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify existing outputs without writes")
    args = parser.parse_args()
    rows, metadata = build(HERE)
    query_path = HERE / "query_NP_Flow.csv"
    metadata_path = HERE / "NP_query_metadata.json"
    report_path = HERE / "NP_QUERY_VALIDATION.md"
    existing_labels(query_path, rows)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    report = report_text(metadata)
    if args.check:
        header, actual = read_csv(query_path)
        require(header == FIELDS and actual == rows, "NP CSV differs from regenerated queries")
        check_metadata(json.loads(metadata_path.read_text()), metadata)
        require(report_path.read_text() == report, "validation report is stale")
    else:
        query_path.write_text(stream.getvalue(), encoding="utf-8")
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_path.write_text(report, encoding="utf-8")
    for r in metadata["instances"]:
        print(f"q{r['source_ca_row_1based']:02d}: flights={r['n_original_flights']}"
              f"+{r['n_added_flights']}={r['n_flights']}  vars={r['n_variables']}"
              f"  B={r['B']}  Z={r['Z']}  OK")
    print("PASS: 15 queries; schema, source alignment, data joins and circulation verified.")
    print("Check only: no files written." if args.check else f"Wrote outputs in {HERE}")


if __name__ == "__main__":
    main()
