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
products = []
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
    products.append(prod)
    revenue[prod] = rev
    demand[prod] = dem
    inventory[prod] = inv

# Validate dimensions
if not (set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()) == set(products)):
    raise ValueError("Mismatch in product keys among revenue, demand, and inventory.")

# Build model
m = gp.Model('DairyOrderFulfillment')
x = m.addVars(products, lb=0, ub=[min(demand[i], inventory[i]) for i in products], vtype=GRB.INTEGER, name='x')

# Objective
m.setObjective(gp.quicksum(revenue[i] * x[i] for i in products), GRB.MAXIMIZE)

# Constraints: x_i <= I_i and x_i <= d_i (already enforced by ub, but add for clarity)
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
