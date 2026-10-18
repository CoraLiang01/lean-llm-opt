LEGACY_OBSERVATION = '[\n  {\n    "source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv",\n    "values": {\n      "Product Name": "Aalopuri",\n      "Revenue": "20",\n      "Demand": "1483",\n      "Initial Inventory": "10440.0"\n    }\n  }\n]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv', 'values': {'Product Name': 'Aalopuri', 'Revenue': '20', 'Demand': '1483', 'Initial Inventory': '10440.0'}}]
import gurobipy as gp
from gurobipy import GRB
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv', 'values': {'Product Name': 'Aalopuri', 'Revenue': '20', 'Demand': '1483', 'Initial Inventory': '10440.0'}}]
aalop_products = []
revenue = {}
demand = {}
initial_inventory = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    pname = vals['Product Name']
    aalop_products.append(pname)
    try:
        revenue[pname] = float(vals['Revenue'])
        demand[pname] = int(float(vals['Demand']))
        initial_inventory[pname] = float(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data for product {pname}: {e}')
for pname in aalop_products:
    if pname not in revenue or pname not in demand or pname not in initial_inventory:
        raise ValueError(f'Missing data for Aalop product {pname}')
m = gp.Model('Aalop_Inventory_Fulfillment')
x_vars = m.addVars(aalop_products, lb=0, ub={p: min(demand[p], initial_inventory[p]) for p in aalop_products}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in aalop_products)), GRB.MAXIMIZE)
for p in aalop_products:
    m.addConstr(x_vars[p] >= 0, name=f'nonneg_{p}')
    m.addConstr(x_vars[p] <= demand[p], name=f'demand_{p}')
    m.addConstr(x_vars[p] <= initial_inventory[p], name=f'inventory_{p}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')