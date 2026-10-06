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
products = []
revenue = {}
demand = {}
inventory = {}

for rec in records:
    vals = rec["values"]
    pname = vals["Product Name"]
    try:
        A_i = float(vals["Revenue"])
        d_i = int(vals["Demand"])
        I_i = int(vals["Initial Inventory"])
    except Exception as e:
        raise ValueError(f"Invalid data for product '{pname}': {e}")
    products.append(pname)
    revenue[pname] = A_i
    demand[pname] = d_i
    inventory[pname] = I_i

# Validate dimensions
if not (set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()) == set(products)):
    raise ValueError("Mismatch in product identifiers among parameters.")

# Build model
m = gp.Model('RetailRevenueMax')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='x')

# Objective
m.setObjective(gp.quicksum(revenue[i] * x[i] for i in products), GRB.MAXIMIZE)

# Constraints
m.addConstrs((x[i] <= inventory[i] for i in products), name='inv')
m.addConstrs((x[i] <= demand[i] for i in products), name='dem')

# Set MIPGap
m.Params.MIPGap = 1e-4

m.optimize()

if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')