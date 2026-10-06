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
items = []
revenue = {}
demand = {}
inventory = {}

for rec in records:
    vals = rec["values"]
    prod = vals["Full_Product_Name"]
    try:
        rev = float(vals["Revenue"])
        dem = int(vals["Demand"])
        inv = int(vals["Initial Inventory"])
    except Exception as e:
        raise ValueError(f"Invalid data for product '{prod}': {e}")
    items.append(prod)
    revenue[prod] = rev
    demand[prod] = dem
    inventory[prod] = inv

# Validate that all required data is present for each item
for prod in items:
    if prod not in revenue or prod not in demand or prod not in inventory:
        raise ValueError(f"Missing data for product '{prod}'.")

m = gp.Model('DairyOrderFulfillment')
x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='x')

m.setObjective(gp.quicksum(revenue[i] * x[i] for i in items), GRB.MAXIMIZE)

m.addConstrs((x[i] <= inventory[i] for i in items), name='')
m.addConstrs((x[i] <= demand[i] for i in items), name='')

m.Params.MIPGap = 1e-4
m.optimize()

if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')