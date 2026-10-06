from gurobipy import Model, GRB

# Extract data from CSVQA_DATA
table = None
for t in CSVQA_DATA["tables"]:
    if t["table_id"] == "file_0_view_0":
        table = t
        break
if table is None:
    raise RuntimeError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")

records = table["records"]

# Build index set and parameter dictionaries
P = []
demand = {}
inv = {}
revenue = {}

for rec in records:
    vals = rec["values"]
    prod = vals["Full_Product_Name"]
    try:
        d = float(vals["Demand"])
        v = float(vals["Initial Inventory"])
        r = float(vals["Revenue"])
    except Exception as e:
        raise RuntimeError(f"Invalid data for product {prod}: {e}")
    P.append(prod)
    demand[prod] = d
    inv[prod] = v
    revenue[prod] = r

# Validate all required data present
for prod in P:
    if prod not in demand or prod not in inv or prod not in revenue:
        raise RuntimeError(f"Missing data for product {prod}")

# Build model
m = Model()
m.Params.MIPGap = 1e-4

# Decision variables: x_i >= 0, continuous
x = m.addVars(P, lb=0, vtype=GRB.CONTINUOUS, name='',)

# Objective: maximize total revenue
m.setObjective(
    sum(revenue[i] * x[i] for i in P),
    GRB.MAXIMIZE
)

# Constraints
for i in P:
    m.addConstr(x[i] <= inv[i], name='')
    m.addConstr(x[i] <= demand[i], name='')

# Optimize
m.optimize()

if m.Status == GRB.OPTIMAL:
    print(f"ObjVal {m.ObjVal}")
    for i in P:
        print(f"{x[i].VarName} {x[i].X}")
else:
    print(f"Solver status: {m.Status}")