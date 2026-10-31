"""CA protocol v1: exact five-file tools and unmodified, complete LLM output."""
from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

from Test_Dataset.Air_NRM.large_scale.np_lean_workflow import (
    AirNrmLargeScaleData, ParsedNPQuery, PRODUCTS, FLIGHT_RE, CONSUME_RE,
    HAIRCUT_RE, finite_number, normalize_time, require, ANSWER_FENCE_RE,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Test_Dataset/Air_NRM/large_scale"
PARAMETER_ORDER = ("products", "flight_info", "avg_price", "value_list", "ratio_list",
                   "value_0_list", "ratio_0_list", "avg_pax", "capacity_consum", "flight_capacity")
PROTOCOL = "ca-direct-full-v1"


def parse_ca_query(query: str) -> ParsedNPQuery:
    """Reuse the typed common fields; z=0 is unused and never enters a CA model."""
    require("flow conservation" not in query.lower(), "CA cannot run an NP/Flow query")
    require(not re.search(r"recommend\s+at\s+most|cardinality", query, re.I),
            "CA cannot run a flight-selection query")
    flights = tuple((a, b, normalize_time(t)) for a, b, t in FLIGHT_RE.findall(query))
    require(bool(flights) and len(flights) == len(re.findall(r"Departure\s+Time\s*=", query, re.I)),
            "missing/malformed candidate flight")
    require(len(set(flights)) == len(flights), "duplicate candidate flight")
    pairs = CONSUME_RE.findall(query)
    consumption = {name: finite_number(v, name, positive=True) for name, v in pairs}
    require(len(pairs) == 4 and set(consumption) == set(PRODUCTS), "four fare consumptions required")
    fraction = HAIRCUT_RE.findall(query)
    require(len(fraction) == 1, "one weekly-capacity percentage is required")
    fraction = finite_number(fraction[0], "capacity fraction", positive=True) / 100
    require(fraction <= 1, "capacity fraction exceeds one")
    return ParsedNPQuery(flights, consumption, fraction, 0)


class AirNrmCAData(AirNrmLargeScaleData):
    """Share CSV indexing/validation; CA emits no selection or flow parameter."""

    def assemble(self, query: str) -> dict:
        parsed = parse_ca_query(query)
        info = dict(enumerate(parsed.flights))
        p = {name: {} for name in PARAMETER_ORDER}
        p["products"] = list(PRODUCTS)
        p["flight_info"] = info
        p["capacity_consum"] = dict(enumerate(parsed.consumption[name] for name in PRODUCTS))
        for fid, (origin, destination, departure) in info.items():
            od = (origin, destination)
            require(od in self._demand, f"missing demand {od}")
            weekly = self._capacity.get((origin, destination, departure))
            require(weekly is not None and weekly > 0, f"missing positive capacity {info[fid]}")
            p["flight_capacity"][fid] = round(weekly * parsed.capacity_fraction, 10)
            values, ratios, value_0, ratio_0 = self._coefficients(od, departure)
            p["value_0_list"][od], p["ratio_0_list"][od] = value_0, ratio_0
            p["avg_pax"][od] = finite_number(self._demand[od]["Avg Pax"], f"demand {od}", positive=True)
            for pid, product in enumerate(PRODUCTS):
                key = (origin, destination, departure, product)
                require(key in self._price, f"missing price {key}")
                p["avg_price"][fid, pid] = self._price[key]
                p["value_list"][fid, pid] = values[product]
                p["ratio_list"][fid, pid] = ratios[product]
        # Reuse checks for common fields. The temporary z=0 is not a model input.
        self.validate_assembled(parsed, {**p, "option_num": 0})
        return p


def format_parameters(parameters: dict) -> str:
    return "\n".join(f"{name}={parameters[name]!r}" for name in PARAMETER_ORDER) + "\n"


def validate_complete_generated_answer(answer: str, parameters: dict) -> str:
    """Reject the old failure modes; extract code without changing any character."""
    require(isinstance(answer, str) and bool(answer.strip()), "empty model answer")
    require("Agent stopped due to" not in answer, "agent stopped before producing an answer")
    positions = [answer.find(h) for h in ("### Objective Function", "### Constraints", "### Generated Code")]
    require(0 <= positions[0] < positions[1] < positions[2], "missing/incorrectly ordered answer headings")
    require("### Retrieved Information" not in answer, "duplicate parameter section is not allowed")
    model_text = answer[:positions[2]].lower()
    require(all(term in model_text for term in ("variable", "capacity", "balance", "scale")),
            "mathematical explanation is incomplete")
    blocks = [m for m in ANSWER_FENCE_RE.finditer(answer) if m.group(1).lower() in ("python", "py")]
    require(len(blocks) == 1 and blocks[0].start() > positions[2], "one tagged Python block is required")
    code = blocks[0].group(2).strip()
    require("gurobipy" in code and "Model(" in code, "missing Gurobi model")
    require("..." not in code, "ellipsis placeholder in generated code")
    tree = ast.parse(code)
    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in PARAMETER_ORDER:
                require(name not in assignments, f"duplicate parameter assignment: {name}")
                try:
                    assignments[name] = ast.literal_eval(node.value)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"parameter {name} must be a complete literal") from exc
    missing = [name for name in PARAMETER_ORDER if name not in assignments]
    require(not missing, f"missing parameters: {missing}")
    changed = [name for name in PARAMETER_ORDER if assignments[name] != parameters[name]]
    require(not changed, f"changed parameters: {changed}")
    # Later writes to the parameter names would invalidate the literal check.
    for node in ast.walk(tree):
        if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            target = node.target
            require(not isinstance(target, ast.Name) or target.id not in PARAMETER_ORDER,
                    "parameter reassignment/annotation is not allowed")
    return code


EXAMPLE_MODEL_CODE = '''import gurobipy as gp
from gurobipy import GRB
model = gp.Model("air_nrm_ca")
x = model.addVars(avg_price.keys(), lb=0.0, name="x")
x_o = model.addVars(avg_pax.keys(), lb=0.0, name="x_o")
model.setObjective(gp.quicksum(avg_price[c] * x[c] for c in avg_price), GRB.MAXIMIZE)
for fid in flight_info:
    model.addConstr(gp.quicksum(capacity_consum[pid] * x[fid,pid]
                               for pid in range(len(products))) <= flight_capacity[fid])
for od in avg_pax:
    model.addConstr(ratio_0_list[od] * x_o[od] + gp.quicksum(
        ratio_list[fid,pid] * x[fid,pid] for fid, info in flight_info.items()
        if info[:2] == od for pid in range(len(products))) == avg_pax[od])
for fid, info in flight_info.items():
    od = info[:2]
    for pid in range(len(products)):
        model.addConstr(value_0_list[od] * x[fid,pid] <= value_list[fid,pid] * x_o[od])
model.optimize()
if model.status == GRB.OPTIMAL:
    print("Optimal objective value:", model.objVal)
else:
    print("No optimal solution found; status:", model.status)
'''

MODEL_EXPLANATION = '''### Objective Function
Maximize total expected ticket-sales revenue: sum(avg_price[fid,pid] * x[fid,pid]).
### Constraints
Decision variables x[fid,pid] and x_o[od] are continuous and nonnegative.
Capacity: for each flight, sum(capacity_consum[pid] * x[fid,pid]) <= flight_capacity[fid].
Balance: for each OD, ratio_0_list[od] * x_o[od] plus ratio-weighted sales equals avg_pax[od].
Scale: value_0_list[od] * x[fid,pid] <= value_list[fid,pid] * x_o[od].
This CA problem has no binary flight-selection or airport flow constraints.
### Generated Code
'''


def example_answer(parameters: dict) -> str:
    return MODEL_EXPLANATION + "```python\n" + format_parameters(parameters) + EXAMPLE_MODEL_CODE + "\n```"


def install_ca_workflow(namespace: dict) -> dict:
    data = AirNrmCAData(*namespace["LoadFiles"]())
    state = {"query": "", "retriever": None}
    example_path = ROOT / "Large_Scale_Or_Files/RAG_Example_SBLP_CA_LargeScale.csv"
    with example_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1 and set(rows[0]) == {"prompt"}, "expected one label-free CA exemplar")
    example_query = rows[0]["prompt"]
    data.assemble(example_query)

    def set_query(query):
        parse_ca_query(query)
        state["query"] = query

    def csvqa(candidate):
        require(bool(state["query"]), "current query was not set")
        return format_parameters(data.assemble(state["query"]))

    def ca_agent(query):
        set_query(query)
        if state["retriever"] is None:
            loader = namespace["CSVLoader"](file_path=str(example_path), encoding="utf-8")
            vectors = namespace["FAISS"].from_documents(loader.load(), namespace["EMBEDDINGS"])
            state["retriever"] = namespace["lx"].build_retriever(namespace["CFG"], vectors, "air_examples")
        docs = state["retriever"].invoke(query)
        require(bool(docs), "CA example retrieval returned no result")
        retrieved = docs[0].page_content.replace("prompt:", "", 1).strip()
        require(parse_ca_query(retrieved) == parse_ca_query(example_query), "unexpected CA exemplar")
        exemplar = ("Question: " + retrieved + "\nThought: I need every current-query parameter.\n"
                    "Action: CSVQA\nAction Input: " + retrieved + "\nObservation:\n"
                    + format_parameters(data.assemble(retrieved))
                    + "\nThought: I will return the complete model and executable code.\nFinal Answer:\n"
                    + example_answer(data.assemble(retrieved)))
        exemplar = exemplar.replace("{", "{{").replace("}", "}}")
        prefix = '''You generate a large-scale Air-NRM CA SBLP formulation and complete Gurobi code.
Call CSVQA with the complete current query before returning a Final Answer.
The example illustrates the format; its parameters must never be reused for the test query.
Follow the complete worked example below:
''' + exemplar + '''

For the CURRENT query, your Final Answer must have Objective Function, Constraints, and
Generated Code headings exactly as shown. Define the decision variables and explain
capacity, balance, scale and nonnegativity. This is CA: no NP binary/cardinality/flow model.
Use exactly one tagged Python code block. At its top copy all TEN CSVQA assignments:
products, flight_info, avg_price, value_list, ratio_list, value_0_list, ratio_0_list,
avg_pax, capacity_consum, flight_capacity. EACH assignment MUST start on a NEW LINE.
After a closing dictionary brace, output a newline before the next parameter name.
Copy all keys and numbers exactly, with no omissions, placeholders, ellipses, rounding,
Observation: labels or reference to external files. Append the full Gurobi model in that
same Python block. Do not duplicate a Retrieved Information section. Return every
assignment, not just avg_price. Do not put Thought/Action/Observation text inside code.
'''
        tools = [namespace["Tool"](name="CSVQA", func=csvqa,
                 description="Retrieve all ten exact parameter assignments for the complete current CA query.")]
        llm = namespace["lx"].build_llm(namespace["CFG"], "data_agent")
        return namespace["initialize_agent"](
            tools, llm=llm, agent=namespace["AgentType"].ZERO_SHOT_REACT_DESCRIPTION,
            **namespace["lx"].agent_kwargs(namespace["CFG"], prefix,
                                           "Begin!\nQuestion: {input}\n{agent_scratchpad}"))

    def process_ca(query):
        # The batch records the raw string before validation, even on failure.
        return ca_agent(query).invoke({"input": query})["output"]

    return {"CA_DATA": data, "CA_SET_CURRENT_QUERY": set_query,
            "CA_Agent": ca_agent, "ProcessCA": process_ca}
