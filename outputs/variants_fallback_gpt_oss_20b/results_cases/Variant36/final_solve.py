import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# ------------------------------------------------------------------
# 1. Load the prepared data
# ------------------------------------------------------------------
DATA_PATH = "work/Variant36/prepared_inputs.json"

# The JSON contains a list of tables under the key 'tables'
data_json = pd.read_json(DATA_PATH, typ='series')
tables = {t['name']: t['rows'] for t in data_json['tables']}

# ------------------------------------------------------------------
# 2. Build sets and parameter dictionaries
# ------------------------------------------------------------------
# Items
item_rows = tables['item']
I = [row['item_ref'] for row in item_rows]

# Helper to convert numeric fields that may be strings
def to_int(val):
    if isinstance(val, str):
        # Remove commas if present
        val = val.replace(',', '')
    return int(val)

auth = {row['item_ref']: to_int(row['authorized']) for row in item_rows}
min_lot = {row['item_ref']: to_int(row['minimum_lot']) for row in item_rows}
max_order = {row['item_ref']: to_int(row['maximum_order']) for row in item_rows}
a = {row['item_ref']: to_int(row['unit_benefit_cents']) for row in item_rows}
f = {row['item_ref']: to_int(row['item_fee_cents']) for row in item_rows}
cat_of_item = {row['item_ref']: row['category'] for row in item_rows}

# Categories
cat_rows = tables['category']
C = [row['category'] for row in cat_rows]
c_min = {row['category']: to_int(row['minimum_quantity']) for row in cat_rows}
c_max = {row['category']: to_int(row['maximum_quantity']) for row in cat_rows}
F = {row['category']: to_int(row['activation_fee_cents']) for row in cat_rows}

# Resources
usage_rows = tables['usage']
u = {}
for r in usage_rows:
    key = (r['item_ref'], r['resource'])
    u[key] = to_int(r['amount'])

# Capacity ledger
cap_rows = tables['capacity_ledger']
L = {}
for r in cap_rows:
    L[r['resource']] = L.get(r['resource'], 0) + to_int(r['amount'])
R = list(L.keys())

# Bundles
bundle_rows = tables['bundle']
B = list(range(len(bundle_rows)))  # integer indices
bundle_items = {}
bonus = {}
for idx, row in enumerate(bundle_rows):
    bundle_items[idx] = (row['item_a'], row['item_b'])
    bonus[idx] = to_int(row['bonus_cents'])

# Incompatibilities
incomp_rows = tables['incompatible']
P = [(row['item_a'], row['item_b']) for row in incomp_rows]

# Requires
requires_rows = tables['requires']
K = [(row['item_ref'], row['prerequisite_ref']) for row in requires_rows]

# ------------------------------------------------------------------
# 3. Create the Gurobi model
# ------------------------------------------------------------------
model = gp.Model("air_conditioner_placement")
model.Params.OutputFlag = 0  # silence Gurobi output

# Decision variables
q = model.addVars(I, vtype=GRB.INTEGER, name="q")
z = model.addVars(I, vtype=GRB.BINARY, name="z")
w = model.addVars(C, vtype=GRB.BINARY, name="w")
y = model.addVars(B, vtype=GRB.BINARY, name="y")

# ------------------------------------------------------------------
# 4. Constraints
# ------------------------------------------------------------------
# 4.1 Item bounds
for i in I:
    model.addConstr(q[i] >= min_lot[i] * z[i], name=f"min_lot_{i}")
    model.addConstr(q[i] <= max_order[i] * z[i], name=f"max_order_{i}")
    model.addConstr(z[i] <= auth[i], name=f"auth_{i}")

# 4.2 Category limits
for c in C:
    items_in_c = [i for i in I if cat_of_item[i] == c]
    model.addConstr(gp.quicksum(q[i] for i in items_in_c) >= c_min[c] * w[c], name=f"cat_min_{c}")
    model.addConstr(gp.quicksum(q[i] for i in items_in_c) <= c_max[c] * w[c], name=f"cat_max_{c}")
    for i in items_in_c:
        model.addConstr(w[c] >= z[i], name=f"cat_use_{c}_{i}")

# 4.3 Resource capacity
for r in R:
    model.addConstr(gp.quicksum(u.get((i, r), 0) * q[i] for i in I) <= L[r], name=f"res_cap_{r}")

# 4.4 Incompatibilities
for (i, j) in P:
    model.addConstr(z[i] + z[j] <= 1, name=f"incomp_{i}_{j}")

# 4.5 Requires
for (i, p) in K:
    model.addConstr(z[i] <= z[p], name=f"requires_{i}_{p}")

# 4.6 Bundle activation
for k in B:
    i_a, i_b = bundle_items[k]
    model.addConstr(y[k] <= z[i_a], name=f"bundle_a_{k}")
    model.addConstr(y[k] <= z[i_b], name=f"bundle_b_{k}")
    model.addConstr(y[k] >= z[i_a] + z[i_b] - 1, name=f"bundle_act_{k}")

# ------------------------------------------------------------------
# 5. Objective
# ------------------------------------------------------------------
obj = (
    gp.quicksum(a[i] * q[i] for i in I)
    - gp.quicksum(f[i] * z[i] for i in I)
    - gp.quicksum(F[c] * w[c] for c in C)
    + gp.quicksum(bonus[k] * y[k] for k in B)
)
model.setObjective(obj, GRB.MAXIMIZE)

# ------------------------------------------------------------------
# 6. Solve
# ------------------------------------------------------------------
model.optimize()

# ------------------------------------------------------------------
# 7. Output
# ------------------------------------------------------------------
if model.status == GRB.OPTIMAL:
    print(f"Optimal net benefit (cents): {int(model.objVal)}")
    selected = [(i, int(q[i].X)) for i in I if q[i].X > 0]
    print("Selected items and quantities:")
    for i, qty in selected:
        print(f"  {i}: {qty}")
else:
    print(f"Solver status: {model.status}")