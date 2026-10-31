#!/usr/bin/env python3
"""Small-scale Air-NRM NP/Flow workflow.

This is the small-scale counterpart of ``large_scale/np_lean_workflow.py``.
It deliberately uses the same integer flight/product key contract so the LLM
never has to manufacture keys such as ``(BA,06:25,f)`` by string splicing.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path

from Test_Dataset.Air_NRM.large_scale.np_lean_workflow import (
    PARAMETER_ORDER,
    format_parameters,
    require,
    validate_complete_generated_answer,
)


HERE = Path(__file__).resolve().parent
PRODUCTS = ("Eco_flexi", "Eco_lite")
FLIGHT_RE = re.compile(
    r"\(OD\s*=\s*\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)\s*AND\s+"
    r"Departure\s*Time\s*=\s*'(\d{1,2}:\d{2})'\)"
)
CONSUME_RE = re.compile(
    r"each\s+(Eco_\w+)\s+ticket\s+consumes\s+([\d.]+)\s+unit", re.I
)
CARDINALITY_RE = re.compile(r"optimal\s+(\d+)\s+flights", re.I)


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
    z: int


def parse_np_query(query: str) -> ParsedNPQuery:
    flights = tuple(
        (origin, destination, normalize_time(departure))
        for origin, destination, departure in FLIGHT_RE.findall(str(query))
    )
    stated = len(re.findall(r"Departure\s+Time\s*=", str(query), re.I))
    require(flights and len(flights) == stated, "missing or malformed flight clause")
    require(len(flights) == len(set(flights)), "duplicate candidate flight")

    pairs = CONSUME_RE.findall(str(query))
    consumption = {
        product: finite_number(value, product, positive=True)
        for product, value in pairs
    }
    # The query says Eco_flex while the CSV product is Eco_flexi.
    if "Eco_flex" in consumption:
        consumption["Eco_flexi"] = consumption.pop("Eco_flex")
    require(len(pairs) == len(PRODUCTS), "capacity consumption is repeated or incomplete")
    require(set(consumption) == set(PRODUCTS),
            f"expected {list(PRODUCTS)}, got {sorted(consumption)}")

    z_values = [int(value) for value in CARDINALITY_RE.findall(str(query))]
    require(len(z_values) == 1 and z_values[0] > 0,
            f"query must state one positive flight limit, got {z_values}")
    return ParsedNPQuery(flights, consumption, z_values[0])


class AirNrmSmallScaleData:
    """Deterministic four-file join with stable integer model keys."""

    def __init__(self, v1, v2, demand, flight):
        self.v1 = v1.copy()
        self.v2 = v2.copy()
        self.demand = demand.copy()
        self.flight = flight.copy()

        self._price = {}
        for row in self.flight.to_dict("records"):
            od = parse_od(row["Oneway_OD"])
            key = (*od, normalize_time(row["Departure Time"]), row["Oneway_Product"])
            require(key not in self._price, f"duplicate price row {key}")
            self._price[key] = finite_number(row["Avg Price"], f"price {key}", positive=True)

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
        )

    def _coefficients(self, od: tuple[str, str], departure: str):
        require(od in self._v1 and od in self._v2, f"missing v1/v2 row for {od}")
        window = time_window(departure)
        values, ratios = {}, {}
        for product in PRODUCTS:
            column = product + "*" + window
            require(column in self._v1[od] and column in self._v2[od],
                    f"missing attraction column {column}")
            values[product] = finite_number(self._v1[od][column], f"v1 {od} {column}", positive=True)
            ratios[product] = finite_number(self._v2[od][column], f"v2 {od} {column}")
        value_0 = finite_number(self._v1[od]["no_purchase"], f"v1 no_purchase {od}", positive=True)
        ratio_0 = finite_number(self._v2[od]["no_purchase"], f"v2 no_purchase {od}", positive=True)
        return values, ratios, value_0, ratio_0

    def assemble(self, query: str) -> dict:
        parsed = parse_np_query(query)
        flight_info = {fid: flight for fid, flight in enumerate(parsed.flights)}
        avg_price, value_list, ratio_list = {}, {}, {}
        value_0_list, ratio_0_list, avg_pax = {}, {}, {}

        for fid, (origin, destination, departure) in flight_info.items():
            od = (origin, destination)
            require(od in self._demand, f"missing demand row for {od}")
            values, ratios, value_0, ratio_0 = self._coefficients(od, departure)
            for pid, product in enumerate(PRODUCTS):
                cell = (fid, pid)
                price_key = (origin, destination, departure, product)
                require(price_key in self._price, f"missing price {price_key}")
                avg_price[cell] = self._price[price_key]
                value_list[cell] = values[product]
                ratio_list[cell] = ratios[product]
            value_0_list[od] = value_0
            ratio_0_list[od] = ratio_0
            avg_pax[od] = finite_number(self._demand[od]["Avg Pax"], f"demand {od}", positive=True)

        parameters = {
            "products": list(PRODUCTS),
            "flight_info": flight_info,
            "avg_price": avg_price,
            "value_list": value_list,
            "ratio_list": ratio_list,
            "value_0_list": value_0_list,
            "ratio_0_list": ratio_0_list,
            "avg_pax": avg_pax,
            "capacity_consum": {pid: parsed.consumption[product]
                                 for pid, product in enumerate(PRODUCTS)},
            "flight_capacity": {fid: 187.0 for fid in flight_info},
            "option_num": parsed.z,
        }
        self.validate_assembled(parsed, parameters)
        return parameters

    @staticmethod
    def validate_assembled(parsed: ParsedNPQuery, parameters: dict) -> None:
        n_flights = len(parsed.flights)
        n_ods = len({flight[:2] for flight in parsed.flights})
        cells = len(PRODUCTS) * n_flights
        require(len(parameters["flight_info"]) == n_flights, "flight_info size mismatch")
        for name in ("avg_price", "value_list", "ratio_list"):
            require(len(parameters[name]) == cells, f"{name} size mismatch")
            require(set(parameters[name]) == set(parameters["avg_price"]),
                    f"{name} cell keys differ")
        for name in ("value_0_list", "ratio_0_list", "avg_pax"):
            require(len(parameters[name]) == n_ods, f"{name} OD size mismatch")
        require(parameters["option_num"] == parsed.z, "Z was not preserved")


SMALL_EXAMPLE_MODEL_CODE = r'''import gurobipy as gp
from gurobipy import GRB
# All exact CSVQA parameter assignments appear above this model code.

M = 10000000
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
        model.addConstr(
            x[fid, pid] <= M * y[fid],
            name="big_m_" + str(fid) + "_" + str(pid),
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


EXAMPLE_QUERY = (
    "Considering that each Eco_flex ticket consumes 1.6 units of flight capacity "
    "and each Eco_lite ticket consumes 1 unit of capacity, while enforcing flow "
    "conservation constraints at each airport, develop an SBLP model. Recommend "
    "the optimal 4 flights among: "
    "(OD = ('A', 'B') AND Departure Time='11:20'), "
    "(OD = ('B', 'A') AND Departure Time='09:05'), "
    "(OD = ('A', 'C') AND Departure Time='09:45'), "
    "(OD = ('C', 'A') AND Departure Time='07:40')."
)


def install_np_workflow(namespace: dict) -> dict:
    """Install the dedicated small-scale NP tools into an initialized notebook."""
    required = (
        "CFG", "lx", "LoadFiles", "Tool", "initialize_agent", "AgentType",
        "Document", "FAISS", "EMBEDDINGS",
    )
    missing = [name for name in required if name not in namespace]
    require(not missing, f"notebook setup missing {missing}")
    v1, v2, demand, flight = namespace["LoadFiles"]()
    data = AirNrmSmallScaleData(v1, v2, demand, flight)
    state = {"current_query": "", "current_parameters": None}

    def authoritative(candidate: str) -> str:
        current = state["current_query"]
        if not current:
            return candidate
        try:
            return candidate if parse_np_query(candidate) == parse_np_query(current) else current
        except Exception:
            return current

    def set_current_query(query: str) -> None:
        parse_np_query(query)
        state["current_query"] = query
        state["current_parameters"] = None

    def retrieve_key_information(query: str) -> str:
        complete = authoritative(query)
        return (complete + "\n\nCNAME_STATUS: identifiers_only; "
                "MANDATORY_NEXT_ACTION: call CSVQA with this complete query.")

    def assemble_with_retrieval(query: str) -> dict:
        """Retrieve current-query prices and demand through retriever.invoke()."""
        complete_query = authoritative(query)
        parameters = data.assemble(complete_query)

        flight_docs = []
        for row in data.flight.to_dict("records"):
            od = parse_od(row["Oneway_OD"])
            departure = normalize_time(row["Departure Time"])
            product = row["Oneway_Product"]
            price = finite_number(row["Avg Price"], "retrieved price", positive=True)
            flight_docs.append(namespace["Document"](
                page_content=(f"OD={od}, Departure Time={departure}, "
                              f"Oneway_Product={product}, avg_price={price}"),
                metadata={"OD": str(od), "time": departure,
                          "product": product, "avg_price": price},
            ))
        flight_vectors = namespace["FAISS"].from_documents(
            flight_docs, namespace["EMBEDDINGS"]
        )

        for fid, (origin, destination, departure) in parameters["flight_info"].items():
            od_text = str((origin, destination))
            for pid, product in enumerate(parameters["products"]):
                retriever = namespace["lx"].build_retriever(
                    namespace["CFG"], flight_vectors, "air_flight",
                    search_kwargs={"filter": {
                        "OD": od_text, "time": departure, "product": product,
                    }},
                )
                docs = retriever.invoke(
                    f"OD={od_text}, Departure Time={departure}, "
                    f"Oneway_Product={product}, avg_price="
                )
                exact = [doc for doc in docs
                         if doc.metadata.get("OD") == od_text
                         and doc.metadata.get("time") == departure
                         and doc.metadata.get("product") == product]
                require(len(exact) == 1,
                        f"retriever did not return exactly one price for "
                        f"{(origin, destination, departure, product)}")
                parameters["avg_price"][fid, pid] = finite_number(
                    exact[0].metadata["avg_price"], "retrieved avg_price", positive=True
                )

        demand_docs = []
        for row in data.demand.to_dict("records"):
            od = parse_od(row["Oneway_OD"])
            demand_value = finite_number(row["Avg Pax"], "retrieved demand", positive=True)
            demand_docs.append(namespace["Document"](
                page_content=f"OD={od}, avg_pax={demand_value}",
                metadata={"OD": str(od), "avg_pax": demand_value},
            ))
        demand_vectors = namespace["FAISS"].from_documents(
            demand_docs, namespace["EMBEDDINGS"]
        )
        for od in parameters["avg_pax"]:
            od_text = str(od)
            retriever = namespace["lx"].build_retriever(
                namespace["CFG"], demand_vectors, "air_demand",
                search_kwargs={"filter": {"OD": od_text}},
            )
            docs = retriever.invoke(f"OD={od_text}, avg_pax=")
            exact = [doc for doc in docs if doc.metadata.get("OD") == od_text]
            require(len(exact) == 1,
                    f"retriever did not return exactly one demand for {od}")
            parameters["avg_pax"][od] = finite_number(
                exact[0].metadata["avg_pax"], "retrieved avg_pax", positive=True
            )

        data.validate_assembled(parse_np_query(complete_query), parameters)
        return parameters

    def csv_qa_tool_flow(query: str) -> str:
        parameters = assemble_with_retrieval(query)
        state["current_parameters"] = parameters
        return format_parameters(parameters)

    example_parameters = format_parameters(data.assemble(EXAMPLE_QUERY))
    fewshot = f'''Question: {EXAMPLE_QUERY}

Action: CName
Action Input: {EXAMPLE_QUERY}
Observation: identifiers preserved; call CSVQA next.

Action: CSVQA
Action Input: {EXAMPLE_QUERY}
Observation:
{example_parameters}

Final Answer:
### Objective Function
Maximize total expected ticket-sales revenue.

### Constraints
Decision variables; per-flight capacity; OD balance; scale; big-M
sales/selection linking; cardinality; airport flow conservation; domains.

### Generated Code
```python
{example_parameters}
{SMALL_EXAMPLE_MODEL_CODE}
```'''.replace("{", "{{").replace("}", "}}")

    def flow_agent(query: str):
        set_current_query(query)
        tools = [
            namespace["Tool"](
                name="CName",
                func=retrieve_key_information,
                description="Preserve the complete candidate-flight query; then call CSVQA.",
            ),
            namespace["Tool"](
                name="CSVQA",
                func=csv_qa_tool_flow,
                description="Return all exact current-query model parameters.",
            ),
        ]
        llm = namespace["lx"].build_llm(namespace["CFG"], "data_agent")
        prefix = f'''You generate a small-scale Air-NRM NP/Flow model and complete
Gurobi code. Use CName first and CSVQA second.

Follow this label-free example exactly:
{fewshot}

Requirements:
- Emit Objective Function, Constraints, then exactly one Generated Code Python block.
- Copy all 11 CSVQA assignments exactly once and never abbreviate them.
- Use integer flight ids from flight_info and integer product ids; never construct
  ticket keys by concatenating parentheses or strings.
- Add exactly one capacity constraint for every fid in flight_info.
- Use all products in each capacity constraint.
- Link x[fid,pid] to y[fid] with M = 10000000, matching the original
  small-scale NP formulation.
- Build airport flow constraints dynamically from flight_info.
- Preserve option_num exactly and include every candidate flight.
'''
        suffix = """Begin!\n\nUser Description: {input}\n{agent_scratchpad}"""
        return namespace["initialize_agent"](
            tools, llm=llm,
            agent=namespace["AgentType"].ZERO_SHOT_REACT_DESCRIPTION,
            **namespace["lx"].agent_kwargs(namespace["CFG"], prefix, suffix),
        )

    def policy_sblp_flow_model_code(query: str) -> str:
        answer = flow_agent(query).invoke({"input": query})["output"]
        parameters = state["current_parameters"]
        require(parameters is not None,
                "CSVQA was not called; current-query parameters were not retrieved")
        validate_complete_generated_answer(answer, parameters)
        return answer

    return {
        "NP_DATA": data,
        "NP_SET_CURRENT_QUERY": set_current_query,
        "NP_ASSEMBLE_WITH_RETRIEVAL": assemble_with_retrieval,
        "retrieve_key_information": retrieve_key_information,
        "csv_qa_tool_flow": csv_qa_tool_flow,
        "FlowAgent": flow_agent,
        "policy_sblp_flow_model_code": policy_sblp_flow_model_code,
        "ProcessPolicyFlow": policy_sblp_flow_model_code,
    }
