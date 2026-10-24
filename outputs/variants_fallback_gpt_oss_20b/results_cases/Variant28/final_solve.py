import pandas as pd
import gurobipy
from gurobipy import GRB

# Path to the prepared input JSON
DATA_PATH = "work/Variant28/prepared_inputs.json"

# Conversion factors to base units
UNIT_CONVERSION = {
    "kwh": ("wh", 1000.0),
    "liter": ("ml", 1000.0),
    "hour": ("minute", 60.0),
}

def convert_amount(amount: float, unit: str) -> float:
    """Convert an amount to its base unit if needed."""
    if unit in UNIT_CONVERSION:
        _, factor = UNIT_CONVERSION[unit]
        return amount * factor
    return amount  # already in base unit or unknown unit

# Load the prepared data
try:
    raw = pd.read_json(DATA_PATH, typ="series")
    data = raw.to_dict()
except Exception as e:
    raise RuntimeError(f"Failed to load JSON data: {e}")

# Extract tables safely
tables = {t["name"]: t.get("rows", []) for t in data.get("tables", [])}

# Helper to get table rows with default empty list
def get_rows(name: str):
    return tables.get(name, [])

# Extract tables
item_rows = get_rows("item")
category_rows = get_rows("category")
bundle_rows = get_rows("bundle")
incompatible_rows = get_rows("incompatible")
requires_rows = get_rows("requires")
usage_rows = get_rows("usage")
capacity_rows = get_rows("capacity_ledger")

# Build item parameters (only authorized items)
item_params = {}
item_category = {}
category_items = {}
for rec in item_rows:
    ref = rec.get("item_ref")
    if not ref:
        continue
    if int(rec.get("authorized", 0)) != 1:
        continue  # skip unauthorized items
    item_params[ref] = {
        "authorized": int(rec["authorized"]),
        "category": rec["category"],
        "unit_benefit_cents": int(rec["unit_benefit_cents"]),
        "item_fee_cents": int(rec["item_fee_cents"]),
        "minimum_lot": int(rec["minimum_lot"]),
        "maximum_order": int(rec["maximum_order"]),
    }
    item_category[ref] = rec["category"]
    category_items.setdefault(rec["category"], []).append(ref)

# Build category parameters
category_params = {}
for rec in category_rows:
    cat = rec["category"]
    category_params[cat] = {
        "activation_fee_cents": int(rec["activation_fee_cents"]),
        "minimum_quantity": int(rec["minimum_quantity"]),
        "maximum_quantity": int(rec["maximum_quantity"]),
    }

# Filter bundles to include only authorized items
bundles = []
for rec in bundle_rows:
    a = rec.get("item_a")
    b = rec.get("item_b")
    if a in item_params and b in item_params:
        bundles.append(
            {
                "item_a": a,
                "item_b": b,
                "bonus_cents": int(rec["bonus_cents"]),
            }
        )

# Incompatible and prerequisite pairs (validate existence)
incompatible_pairs = [
    (rec["item_a"], rec["item_b"])
    for rec in incompatible_rows
    if rec["item_a"] in item_params and rec["item_b"] in item_params
]
requires_pairs = [
    (rec["item_ref"], rec["prerequisite_ref"])
    for rec in requires_rows
    if rec["item_ref"] in item_params and rec["prerequisite_ref"] in item_params
]

# Usage per item-resource (converted to base units)
usage = {}
for rec in usage_rows:
    item_ref = rec.get("item_ref")
    resource = rec.get("resource")
    if not item_ref or not resource:
        continue
    amount = float(rec.get("amount", 0.0))
    unit = rec.get("unit", "")
    amount_base = convert_amount(amount, unit)
    usage[(item_ref, resource)] = amount_base

# Capacity per resource (signed sum, converted to base units)
capacity = {}
for rec in capacity_rows:
    resource = rec.get("resource")
    if not resource:
        continue
    amount = float(rec.get("amount", 0.0))
    unit = rec.get("unit", "")
    amount_base = convert_amount(amount, unit)
    capacity[resource] = capacity.get(resource, 0.0) + amount_base

# Build the Gurobi model
m = gurobipy.Model()
m.setParam("MIPGap", 1e-06)

# Decision variables
q = {i: m.addVar(vtype=GRB.INTEGER, name=f"q_{i}") for i in item_params}
y = {i: m.addVar(vtype=GRB.BINARY, name=f"y_{i}") for i in item_params}
z = {c: m.addVar(vtype=GRB.BINARY, name=f"z_{c}") for c in category_params}
w = {}
for idx, b in enumerate(bundles):
    key = (b["item_a"], b["item_b"])
    w[key] = m.addVar(vtype=GRB.BINARY, name=f"w_{idx}")

# Quantity bounds and activation
for i, p in item_params.items():
    m.addConstr(q[i] >= p["minimum_lot"] * y[i])
    m.addConstr(q[i] <= p["maximum_order"] * y[i])

# Category activation constraints
for c, items in category_items.items():
    sum_y = gurobipy.quicksum(y[i] for i in items)
    m.addConstr(z[c] <= sum_y)
    for i in items:
        m.addConstr(y[i] <= z[c])

# Conditional category quantity limits
for c, p in category_params.items():
    items = category_items.get(c, [])
    sum_q = gurobipy.quicksum(q[i] for i in items)
    m.addConstr(sum_q >= p["minimum_quantity"] * z[c])
    m.addConstr(sum_q <= p["maximum_quantity"] * z[c])

# Incompatibility constraints
for a, b in incompatible_pairs:
    m.addConstr(y[a] + y[b] <= 1)

# Prerequisite constraints
for i, prereq in requires_pairs:
    m.addConstr(y[i] <= y[prereq])

# Bundle bonus constraints
for b in bundles:
    a = b["item_a"]
    b_item = b["item_b"]
    key = (a, b_item)
    m.addConstr(w[key] <= y[a])
    m.addConstr(w[key] <= y[b_item])
    m.addConstr(w[key] >= y[a] + y[b_item] - 1)

# Resource capacity constraints
for r, cap in capacity.items():
    expr = gurobipy.quicksum(usage.get((i, r), 0.0) * q[i] for i in item_params)
    m.addConstr(expr <= cap)

# Objective: maximize net benefit
obj = gurobipy.quicksum(item_params[i]["unit_benefit_cents"] * q[i] for i in item_params)
obj -= gurobipy.quicksum(item_params[i]["item_fee_cents"] * y[i] for i in item_params)
obj -= gurobipy.quicksum(category_params[c]["activation_fee_cents"] * z[c] for c in category_params)
obj += gurobipy.quicksum(b["bonus_cents"] * w[(b["item_a"], b["item_b"])] for b in bundles)

m.setObjective(obj, GRB.MAXIMIZE)

# Optimize
m.optimize()