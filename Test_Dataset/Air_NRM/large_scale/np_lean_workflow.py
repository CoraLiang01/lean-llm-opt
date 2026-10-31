#!/usr/bin/env python3
"""Large-scale Air-NRM NP/Flow workflow used by the LEAN notebook.

The data assembly path is deterministic and can be checked without an API.
Only ``install_np_workflow`` constructs LangChain objects or model calls.
"""

from __future__ import annotations

import ast
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
PRODUCTS = ("Eco_flexi", "Eco_standard", "Eco_value", "Eco_lite")

FLIGHT_RE = re.compile(
    r"\(OD\s*=\s*\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)\s*AND\s+"
    r"Departure\s*Time\s*=\s*'(\d{1,2}:\d{2})'\)"
)
CONSUME_RE = re.compile(
    r"each\s+(Eco_\w+)\s+ticket\s+consumes\s+([\d.]+)\s+unit", re.I
)
HAIRCUT_RE = re.compile(
    r"capacity is\s+([\d.]+)%\s+of its weekly economy capacity", re.I
)
CARDINALITY_RE = re.compile(
    r"(?:recommend\s+at\s+most|cardinality\s+constraint\s+allowing\s+at\s+most)"
    r"\s+(\d+)\s+(?:selected\s+)?flights",
    re.I,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalize_time(value: str) -> str:
    hour, minute = map(int, str(value).split(":"))
    require(0 <= hour < 24 and 0 <= minute < 60, f"invalid time {value!r}")
    return f"{hour:02d}:{minute:02d}"


def parse_od(value: str) -> tuple[str, str]:
    od = ast.literal_eval(str(value))
    require(
        isinstance(od, tuple)
        and len(od) == 2
        and all(isinstance(code, str) and code for code in od)
        and od[0] != od[1],
        f"invalid OD {value!r}",
    )
    return od


def time_window(value: str) -> str:
    minutes = int(value[:2]) * 60 + int(value[3:])
    if 480 <= minutes < 720:
        return "(8am~12pm)"
    if 720 <= minutes < 1080:
        return "(12pm~6pm)"
    if 1080 <= minutes < 1320:
        return "(6pm~10pm)"
    return "(10pm~8am)"


def finite_number(value, label: str, *, positive: bool = False) -> float:
    result = float(value)
    require(
        math.isfinite(result) and (result > 0 if positive else result >= 0),
        f"invalid {'positive ' if positive else ''}{label}: {value!r}",
    )
    return result


@dataclass(frozen=True)
class ParsedNPQuery:
    flights: tuple[tuple[str, str, str], ...]
    consumption: dict[str, float]
    capacity_fraction: float
    z: int


def parse_np_query(query: str) -> ParsedNPQuery:
    """Parse and strictly validate the model-defining NP query fields."""
    flights = tuple(
        (origin, destination, normalize_time(departure))
        for origin, destination, departure in FLIGHT_RE.findall(str(query))
    )
    stated_flights = len(re.findall(r"Departure\s+Time\s*=", str(query), re.I))
    require(flights and len(flights) == stated_flights, "missing or malformed flight clause")
    require(len(flights) == len(set(flights)), "duplicate candidate flight")

    pairs = CONSUME_RE.findall(str(query))
    consumption = {product: finite_number(value, product, positive=True)
                   for product, value in pairs}
    require(len(pairs) == len(PRODUCTS), "capacity consumption is repeated or incomplete")
    require(set(consumption) == set(PRODUCTS),
            f"expected capacity consumption for {list(PRODUCTS)}, got {sorted(consumption)}")

    haircut = HAIRCUT_RE.findall(str(query))
    require(len(haircut) == 1, "query must state exactly one weekly-capacity percentage")
    capacity_fraction = finite_number(haircut[0], "capacity percentage", positive=True) / 100.0
    require(capacity_fraction <= 1.0, "capacity percentage exceeds 100%")

    z_values = [int(value) for value in CARDINALITY_RE.findall(str(query))]
    require(len(z_values) == 2 and z_values[0] == z_values[1],
            f"Z must occur twice and agree, got {z_values}")
    require(z_values[0] > 0, "Z must be positive")

    return ParsedNPQuery(flights, consumption, capacity_fraction, z_values[0])


class AirNrmLargeScaleData:
    """Exact five-file join shared by notebook execution and offline checks."""

    def __init__(self, v1, v2, demand, flight, capacity):
        self.v1 = v1.copy()
        self.v2 = v2.copy()
        self.demand = demand.copy()
        self.flight = flight.copy()
        self.capacity = capacity.copy()

        v1_products = []
        for column in self.v1.columns:
            if "*" in str(column):
                product = str(column).split("*", 1)[0]
                if product not in v1_products:
                    v1_products.append(product)
        require(tuple(v1_products) == PRODUCTS,
                f"unexpected product order in v1.csv: {v1_products}")
        self.short = {product: product.split("_")[-1][0] for product in PRODUCTS}
        require(len(set(self.short.values())) == len(PRODUCTS), "product short-code collision")

        self._price = {}
        for row in self.flight.to_dict("records"):
            key = (*parse_od(row["Oneway_OD"]),
                   normalize_time(row["Departure Time"]), row["Oneway_Product"])
            require(key not in self._price, f"duplicate price row {key}")
            self._price[key] = finite_number(row["Avg Price"], f"price {key}", positive=True)

        self._capacity = {}
        for row in self.capacity.to_dict("records"):
            key = (*parse_od(row["Oneway_OD"]), normalize_time(row["Departure Time"]))
            require(key not in self._capacity, f"duplicate capacity row {key}")
            self._capacity[key] = finite_number(row["Y Seats/Week"], f"capacity {key}")

        self._demand = self._index_od(self.demand, "Oneway_OD", "od_demand.csv")
        self._v1 = self._index_od(self.v1, "OD Pairs", "v1.csv")
        self._v2 = self._index_od(self.v2, "OD Pairs", "v2.csv")

    @staticmethod
    def _index_od(frame, field: str, name: str):
        result = {}
        for row in frame.to_dict("records"):
            od = parse_od(row[field])
            require(od not in result, f"{name}: duplicate OD {od}")
            result[od] = row
        return result

    @classmethod
    def from_path(cls, base: Path | str = HERE):
        import pandas as pd

        base = Path(base)
        return cls(
            pd.read_csv(base / "v1.csv"),
            pd.read_csv(base / "v2.csv"),
            pd.read_csv(base / "od_demand.csv"),
            pd.read_csv(base / "flight.csv"),
            pd.read_csv(base / "flight_capacity.csv"),
        )

    def _coefficients(self, od: tuple[str, str], departure: str):
        require(od in self._v1 and od in self._v2, f"missing v1/v2 row for {od}")
        window = time_window(departure)
        values, ratios = {}, {}
        for product in PRODUCTS:
            column = product + "*" + window
            require(column in self._v1[od] and column in self._v2[od],
                    f"missing attraction column {column}")
            values[product] = finite_number(
                self._v1[od][column], f"v1 {od} {column}", positive=True
            )
            ratios[product] = finite_number(
                self._v2[od][column], f"v2 {od} {column}"
            )
        value_0 = finite_number(self._v1[od]["no_purchase"], f"v1 no_purchase {od}", positive=True)
        ratio_0 = finite_number(self._v2[od]["no_purchase"], f"v2 no_purchase {od}", positive=True)
        return values, ratios, value_0, ratio_0

    def assemble(self, query: str) -> dict:
        """Return every numeric/model-structure parameter, with compact stable keys."""
        parsed = parse_np_query(query)
        flight_info = {fid: flight for fid, flight in enumerate(parsed.flights)}
        avg_price, value_list, ratio_list = {}, {}, {}
        value_0_list, ratio_0_list, avg_pax = {}, {}, {}
        flight_capacity = {}

        for fid, (origin, destination, departure) in flight_info.items():
            od = (origin, destination)
            require(od in self._demand, f"missing demand row for {od}")
            weekly = self._capacity.get((origin, destination, departure))
            require(weekly is not None and weekly > 0,
                    f"missing positive weekly capacity for {(origin, destination, departure)}")
            flight_capacity[fid] = round(weekly * parsed.capacity_fraction, 10)

            values, ratios, value_0, ratio_0 = self._coefficients(od, departure)
            for product_id, product in enumerate(PRODUCTS):
                cell = (fid, product_id)
                price_key = (origin, destination, departure, product)
                require(price_key in self._price, f"missing price {price_key}")
                avg_price[cell] = self._price[price_key]
                value_list[cell] = values[product]
                ratio_list[cell] = ratios[product]

            if od in value_0_list:
                require(value_0_list[od] == value_0 and ratio_0_list[od] == ratio_0,
                        f"inconsistent no_purchase coefficients for {od}")
            value_0_list[od] = value_0
            ratio_0_list[od] = ratio_0
            avg_pax[od] = finite_number(self._demand[od]["Avg Pax"], f"demand {od}", positive=True)

        product_consumption = {
            product_id: parsed.consumption[product]
            for product_id, product in enumerate(PRODUCTS)
        }
        parameters = {
            "products": list(PRODUCTS),
            "flight_info": flight_info,
            "avg_price": avg_price,
            "value_list": value_list,
            "ratio_list": ratio_list,
            "value_0_list": value_0_list,
            "ratio_0_list": ratio_0_list,
            "avg_pax": avg_pax,
            "capacity_consum": product_consumption,
            "flight_capacity": flight_capacity,
            "option_num": parsed.z,
        }
        self.validate_assembled(parsed, parameters)
        return parameters

    @staticmethod
    def validate_assembled(parsed: ParsedNPQuery, parameters: dict) -> None:
        n_flights = len(parsed.flights)
        n_ods = len({flight[:2] for flight in parsed.flights})
        cells = 4 * n_flights
        require(len(parameters["flight_info"]) == n_flights, "flight_info size mismatch")
        for name in ("avg_price", "value_list", "ratio_list"):
            require(len(parameters[name]) == cells, f"{name} size mismatch")
            require(set(parameters[name]) == set(parameters["avg_price"]),
                    f"{name} cell keys differ")
        for name in ("value_0_list", "ratio_0_list", "avg_pax"):
            require(len(parameters[name]) == n_ods, f"{name} OD size mismatch")
        require(len(parameters["flight_capacity"]) == n_flights, "flight capacity size mismatch")
        require(parameters["option_num"] == parsed.z, "Z was not preserved")
        require(set(parameters["capacity_consum"]) == set(range(4)),
                "capacity consumption product ids mismatch")
        require(all(value > 0 for value in parameters["flight_capacity"].values()),
                "non-positive modeled flight capacity")


PARAMETER_ORDER = (
    "products",
    "flight_info",
    "avg_price",
    "value_list",
    "ratio_list",
    "value_0_list",
    "ratio_0_list",
    "avg_pax",
    "capacity_consum",
    "flight_capacity",
    "option_num",
)


def format_parameters(parameters: dict) -> str:
    return "\n".join(f"{name}={parameters[name]!r}" for name in PARAMETER_ORDER) + "\n"


ANSWER_FENCE_RE = re.compile(
    r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```", re.I | re.S
)


def validate_complete_generated_answer(answer: str, parameters: dict) -> str:
    """Validate, but never alter, the self-contained code emitted by the LLM."""
    text = str(answer)
    objective_heading = text.find("### Objective Function")
    constraints_heading = text.find("### Constraints")
    code_heading = text.find("### Generated Code")
    fence_start = text.find("```", code_heading if code_heading >= 0 else 0)
    require(0 <= objective_heading < constraints_heading < code_heading < fence_start,
            "answer must contain Objective Function, Constraints, then Generated Code")
    require("### Retrieved Information" not in text,
            "do not duplicate parameters in a Retrieved Information section")
    model_section = text[objective_heading:code_heading].strip()
    required_model_terms = ("objective", "variable", "capacity", "balance",
                            "scale", "cardinality", "flow")
    missing_model_terms = [term for term in required_model_terms
                           if term not in model_section.lower()]
    require(not missing_model_terms,
            f"### Model section is incomplete: {missing_model_terms}")
    blocks = [
        (match.group(1).lower(), match.group(2).strip(), match.start())
        for match in ANSWER_FENCE_RE.finditer(text)
    ]
    python_blocks = [block for block in blocks if block[0] in ("python", "py")]
    require(len(python_blocks) == 1,
            f"expected exactly one tagged Python code block, got {len(python_blocks)}")
    _, code, python_fence_start = python_blocks[0]
    require(python_fence_start > code_heading,
            "the tagged Python block must appear under ### Generated Code")
    require("gurobipy" in code and "Model(" in code,
            "generated code contains no Gurobi model")
    require("..." not in code, "generated code contains an ellipsis placeholder")
    tree = ast.parse(code)
    require(
        not any(isinstance(node, ast.Constant) and node.value is Ellipsis
                for node in ast.walk(tree)),
        "generated code contains an ellipsis placeholder",
    )
    assignments = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in PARAMETER_ORDER):
            try:
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"parameter {node.targets[0].id} is not a complete literal"
                ) from exc
    missing = [name for name in PARAMETER_ORDER if name not in assignments]
    require(not missing, f"generated code is missing parameters: {missing}")
    different = [name for name in PARAMETER_ORDER
                 if assignments[name] != parameters[name]]
    require(not different, f"generated code changed parameters: {different}")
    return code


EXAMPLE_MODEL_CODE = r'''import gurobipy as gp
from gurobipy import GRB
# All exact CSVQA parameter assignments appear above this model code.

model = gp.Model("air_nrm_np_flow")
x = model.addVars(avg_price.keys(), lb=0.0, name="x")
x_o = model.addVars(avg_pax.keys(), lb=0.0, name="x_o")
y = model.addVars(flight_info.keys(), vtype=GRB.BINARY, name="y")

model.setObjective(
    gp.quicksum(avg_price[cell] * x[cell] for cell in avg_price),
    GRB.MAXIMIZE,
)

for fid in flight_info:
    model.addConstr(
        gp.quicksum(capacity_consum[pid] * x[fid, pid]
                    for pid in range(len(products))) <= flight_capacity[fid],
        name="capacity_" + str(fid),
    )

for od in avg_pax:
    model.addConstr(
        ratio_0_list[od] * x_o[od]
        + gp.quicksum(ratio_list[fid, pid] * x[fid, pid]
                      for fid, info in flight_info.items() if info[:2] == od
                      for pid in range(len(products))) == avg_pax[od],
        name="balance_" + "_".join(od),
    )

for fid, info in flight_info.items():
    od = info[:2]
    for pid in range(len(products)):
        model.addConstr(
            value_0_list[od] * x[fid, pid]
            <= value_list[fid, pid] * x_o[od],
            name="scale_" + str(fid) + "_" + str(pid),
        )
        # Tight sales/selection link derived from capacity; no arbitrary big-M.
        model.addConstr(
            x[fid, pid]
            <= (flight_capacity[fid] / capacity_consum[pid]) * y[fid],
            name="select_" + str(fid) + "_" + str(pid),
        )

model.addConstr(gp.quicksum(y[fid] for fid in flight_info) <= option_num,
                name="cardinality")

airports = sorted({code for info in flight_info.values() for code in info[:2]})
for airport in airports:
    model.addConstr(
        gp.quicksum(y[fid] for fid, info in flight_info.items()
                    if info[1] == airport)
        == gp.quicksum(y[fid] for fid, info in flight_info.items()
                       if info[0] == airport),
        name="flow_" + airport,
    )

model.optimize()
if model.status == GRB.OPTIMAL:
    print("Optimal objective value:", model.objVal)
    print("Selected flights:",
          [flight_info[fid] for fid in flight_info if y[fid].X > 0.5])
else:
    print("No optimal solution found; status:", model.status)
'''


def _read_example_query(path: Path) -> str:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1 and list(rows[0]) == ["prompt"],
            "Flow exemplar must contain exactly one prompt column and one row")
    parse_np_query(rows[0]["prompt"])
    return rows[0]["prompt"]


def install_np_workflow(namespace: dict, *, ablation: str | None = None) -> dict:
    """Install notebook-facing NP functions into an initialized namespace.

    ``ablation`` keeps the large-scale data/model contract fixed while removing
    exactly one LEAN component:

    * ``rag_only`` keeps current-query CSVQA retrieval and removes the exemplar.
    * ``fewshot_only`` keeps the fixed exemplar and replaces current-query
      retrieval with native API file attachments supplied by the notebook.

    The default remains the full LEAN workflow used by the NP-only notebook.
    """
    require(ablation in (None, "rag_only", "fewshot_only"),
            f"unknown NP ablation mode: {ablation!r}")
    required = (
        "CFG", "lx", "LoadFiles", "CSVLoader", "FAISS", "EMBEDDINGS",
        "Tool", "initialize_agent", "AgentType",
    )
    missing = [name for name in required if name not in namespace]
    require(not missing, f"notebook setup missing {missing}")

    data = None
    if ablation != "fewshot_only":
        v1, v2, demand, flight, capacity = namespace["LoadFiles"]()
        data = AirNrmLargeScaleData(v1, v2, demand, flight, capacity)
    example_path = HERE / "RAG_Example_SBLP_Flow_LargeScale_NP.csv"
    example_query = _read_example_query(example_path)
    state = {"current_query": "", "example_retriever": None}

    def authoritative(candidate: str) -> str:
        current = state["current_query"]
        if not current:
            parse_np_query(candidate)
            return candidate
        try:
            candidate_parsed = parse_np_query(candidate)
            current_parsed = parse_np_query(current)
            if candidate_parsed != current_parsed:
                return current
            return candidate
        except Exception:
            return current

    def set_current_query(query: str) -> None:
        parse_np_query(query)
        state["current_query"] = query

    def cname_payload(complete_query: str) -> str:
        return (
            complete_query
            + "\n\nCNAME_STATUS: identifiers_only; numeric_model_parameters_not_retrieved."
              "\nMANDATORY_NEXT_ACTION: call CSVQA with the complete query above before "
              "writing any Final Answer."
        )

    def retrieve_key_information(query: str) -> str:
        """Keep the complete query and make the required next data step explicit."""
        return cname_payload(authoritative(query))

    def csv_qa_tool_flow(query: str) -> str:
        return format_parameters(data.assemble(authoritative(query)))

    def make_fewshot(retrieved_query: str) -> str:
        if ablation == "fewshot_only":
            return (HERE / "workflow_only_np_example.txt").read_text(encoding="utf-8")
        example_parameters = format_parameters(data.assemble(retrieved_query))
        example_cname = cname_payload(retrieved_query)
        return f'''Question: {retrieved_query}

Thought: I must preserve the complete candidate-flight list, Z, capacity percentage,
and all four capacity-consumption coefficients.
Action: CName
Action Input: {retrieved_query}
Observation: {example_cname}

Thought: I need the exact model parameters from the five Air-NRM CSV files.
Action: CSVQA
Action Input: {retrieved_query}
Observation:
{example_parameters}
Thought: I must copy every assignment from the CSVQA observation exactly once,
then append the complete NP Gurobi model in the same code block.

Final Answer:
### Objective Function
Maximize total expected ticket-sales revenue:

`maximize sum(avg_price[fid,pid] * x[fid,pid])`.

### Constraints

#### 1. Decision Variables
`x[fid,pid] >= 0` is sales of product `pid` on flight `fid`; `x_o[od] >= 0`
is no-purchase demand; `y[fid]` is binary flight selection.

#### 2. Capacity Constraints
For each flight, capacity-weighted sales cannot exceed `flight_capacity[fid]`.

#### 3. Balance Constraints
For each OD, ratio-weighted no-purchase demand plus ratio-weighted ticket sales
equals `avg_pax[od]`.

#### 4. Scale Constraints
For each ticket option, `value_0_list[od] * x[fid,pid]` cannot exceed
`value_list[fid,pid] * x_o[od]`.

#### 5. Sales/Selection Linking
Sales are zero unless the flight is selected, using the tight capacity-derived
upper bound `flight_capacity[fid] / capacity_consum[pid]`.

#### 6. Cardinality Constraint
At most `option_num` flights may be selected.

#### 7. Flow Conservation Constraints
At every airport, the number of selected inbound flights equals the number of
selected outbound flights.

#### 8. Variable Domains
Sales and no-purchase variables are nonnegative; flight selections are binary.

### Generated Code
```python
{example_parameters}
{EXAMPLE_MODEL_CODE}
```'''

    def flow_agent(query: str):
        set_current_query(query)
        fewshot_for_prompt = ""
        if ablation != "rag_only":
            if ablation == "fewshot_only":
                # Static, precomputed exemplar: no example retrieval occurs in this arm.
                retrieved_query = example_query
            else:
                if state["example_retriever"] is None:
                    loader = namespace["CSVLoader"](
                        file_path=str(example_path), encoding="utf-8"
                    )
                    vectors = namespace["FAISS"].from_documents(
                        loader.load(), namespace["EMBEDDINGS"]
                    )
                    state["example_retriever"] = namespace["lx"].build_retriever(
                        namespace["CFG"], vectors, "air_examples"
                    )
                # The full arm retrieves its one isolated, label-free NP exemplar.
                docs = state["example_retriever"].invoke(query)
                require(bool(docs) and "prompt:" in docs[0].page_content,
                        "large-scale NP exemplar retrieval returned no prompt")
                retrieved_query = docs[0].page_content.replace("prompt:", "", 1).strip()
                require(parse_np_query(retrieved_query) == parse_np_query(example_query),
                        "retrieved exemplar differs from the isolated NP exemplar")
            fewshot_for_prompt = make_fewshot(retrieved_query).replace(
                "{", "{{"
            ).replace("}", "}}")

        if ablation == "fewshot_only":
            require("API_FILES" in namespace,
                    "workflow-only NP requires native API file attachments")
            tool_rule = (
                "Read the attached raw CSV files through the API. Derive the current "
                "query's parameters yourself. No custom data tools are available."
            )
        else:
            tools = [
                namespace["Tool"](
                    name="CSVQA",
                    func=csv_qa_tool_flow,
                    description=(
                        "Return every exact price, attraction, ratio, demand, capacity, "
                        "consumption, flight identity, and Z value for the current query."
                    ),
                ),
                namespace["Tool"](
                    name="CName",
                    func=retrieve_key_information,
                    description=(
                        "Preserve the full NP query and its identifiers only. This tool "
                        "does not return numeric model parameters; after CName you must "
                        "call CSVQA before producing a Final Answer."
                    ),
                ),
            ]
            tool_rule = (
                "Use CName first and CSVQA second. The tool sequence is mandatory. "
                "CName contains identifiers but no price, attraction, ratio, demand, "
                "or flight-capacity data. Never write `Final Answer:` after CName; "
                "call CSVQA and use that current-query observation first."
            )

        if ablation == "rag_only":
            example_section = (
                "This is the RAG-only ablation: no worked example is available. "
                "Infer the formulation from the requirements below."
            )
        else:
            example_section = (
                "Follow this fixed, label-free large-scale example:\n\n"
                + fewshot_for_prompt
            )
        prefix = f'''You are an assistant that generates a large-scale Air-NRM
SBLP/NP mathematical model and its self-contained Gurobi Python code.

{tool_rule}

{example_section}

Requirements for the final answer:
- Follow the few-shot example's exact mathematical-model format: `### Objective
  Function`, `### Constraints` with its numbered subsections, and then
  `### Generated Code`.
- The mathematical section must define the objective, decision variables,
  capacity, balance, scale, nonnegativity, sales/selection linking, cardinality,
  binary selection, and airport flow-conservation constraints.
- In the mathematical sections, prefer LaTeX delimiters or inline backticks.
  If you use a display fence there, leave it untyped; only the executable block
  under `### Generated Code` may carry the `python` language tag.
- Use all candidates returned in flight_info. Never truncate, summarize, or use ellipses.
- Preserve option_num exactly; it is Z and the constraint is sum(y) <= option_num.
- Preserve all four capacity_consum values and every per-flight flight_capacity value.
- Link each sales variable to y with flight_capacity[fid] / capacity_consum[pid],
  not an arbitrary constant big-M.
- Build airports dynamically from flight_info. For every airport, selected inbound
  flights must equal selected outbound flights.
- Emit exactly one Python code block containing a complete, self-contained program.
- At the top of that same code block, copy all 11 assignments returned by CSVQA:
  products, flight_info, avg_price, value_list, ratio_list, value_0_list,
  ratio_0_list, avg_pax, capacity_consum, flight_capacity, and option_num.
- Copy every key and value exactly. Never shorten a dictionary, use an ellipsis,
  write "same as above", substitute example data, round a number, or move a
  parameter outside the code block.
- After those exact assignments, append the complete Gurobi model. Do not read
  external files and do not include a benchmark label or known optimum.
- Before returning, verify that the single code block defines all 11 parameter
  names and contains no `...` anywhere.
- Do not create a `Retrieved Information` section and do not print parameter
  dictionaries anywhere outside the single Generated Code block. The CSVQA
  observation already contains them; duplicating them wastes output and invites
  abbreviation. Each parameter must appear exactly once in the final answer.
'''
        if ablation == "fewshot_only":
            from airnrm_api_files import FileInputAgent
            prefix = prefix.replace("{{", "{").replace("}}", "}")
            prefix = prefix.replace("copy all 11 assignments returned by CSVQA:",
                                    "write all 11 assignments derived from the attached files:")
            prefix = prefix.replace("The CSVQA\n  observation already contains them; duplicating them wastes output and invites\n  abbreviation.",
                                    "Avoid duplicating parameters.")
            return FileInputAgent(namespace["API_FILES"], prefix)
        llm = namespace["lx"].build_llm(namespace["CFG"], "data_agent")
        suffix = """
Begin!

User Description: {input}
{agent_scratchpad}"""
        return namespace["initialize_agent"](
            tools,
            llm=llm,
            agent=namespace["AgentType"].ZERO_SHOT_REACT_DESCRIPTION,
            **namespace["lx"].agent_kwargs(namespace["CFG"], prefix, suffix),
        )

    def policy_sblp_flow_model_code(query: str) -> str:
        agent = flow_agent(query)
        result = agent.invoke({"input": query})
        answer = result["output"]
        # The full and RAG-only arms have exact current-query parameters, so a
        # truncated answer is a harness failure. Few-shot-only deliberately has
        # no current-query retrieval; score whatever it actually produces.
        if ablation != "fewshot_only":
            validate_complete_generated_answer(answer, data.assemble(query))
        return answer

    def process_policy_flow(query: str) -> str:
        return policy_sblp_flow_model_code(query)

    return {
        "NP_DATA": data,
        "NP_SET_CURRENT_QUERY": set_current_query,
        "retrieve_key_information": retrieve_key_information,
        "csv_qa_tool_flow": csv_qa_tool_flow,
        "FlowAgent": flow_agent,
        "policy_sblp_flow_model_code": policy_sblp_flow_model_code,
        "ProcessPolicyFlow": process_policy_flow,
    }
