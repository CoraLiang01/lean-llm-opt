import gurobipy as gp
from gurobipy import GRB

# Extract data from CSVQA_DATA
table = None
for t in CSVQA_DATA["tables"]:
    if t["table_id"] == "file_0_view_0":
        table = t
        break
if table is None:
    raise ValueError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")

records = table["records"]

# Build index set and parameter dictionaries
items = []
revenue = {}
demand = {}
inventory = {}

for rec in records:
    vals = rec["values"]
    prod = vals["Product Name"]
    try:
        rev = int(vals["Revenue"])
        dem = int(vals["Demand"])
        inv = int(vals["Initial Inventory"])
    except Exception as e:
        raise ValueError(f"Non-integer value in data for product '{prod}': {e}")
    items.append(prod)
    revenue[prod] = rev
    demand[prod] = dem
    inventory[prod] = inv

# Validate dimensions
if not (set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()) == set(items)):
    raise ValueError("Mismatch in index sets for revenue, demand, or inventory.")

# Build and solve the model
m = gp.Model('Retail_Revenue_Optimization')

x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='x')

m.setObjective(gp.quicksum(revenue[i] * x[i] for i in items), GRB.MAXIMIZE)

m.addConstrs((x[i] <= inventory[i] for i in items), name='inv')
m.addConstrs((x[i] <= demand[i] for i in items), name='dem')

m.Params.MIPGap = 1e-4
m.optimize()

if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')