import gurobipy as gp
from gurobipy import GRB

# Extract data from CSVQA_DATA
table = [r["values"] for r in CSVQA_DATA["tables"][0]["records"]]
if not table:
    raise ValueError("No records found for 'Baby' products in file_0_view_0.")

# Build index set and parameter dictionaries
items = []
revenue = {}
demand = {}
inventory = {}

for row in table:
    product = row["Product Name"]
    try:
        A_i = float(row["Revenue"])
        d_i = int(row["Demand"])
        I_i = int(row["Initial Inventory"])
    except Exception as e:
        raise ValueError(f"Invalid data for product {product}: {e}")
    items.append(product)
    revenue[product] = A_i
    demand[product] = d_i
    inventory[product] = I_i

# Validate dimensions
if not (set(revenue) == set(demand) == set(inventory) == set(items)):
    raise ValueError("Mismatch in index sets for revenue, demand, or inventory.")

m = gp.Model('Baby_Product_Fulfillment')
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