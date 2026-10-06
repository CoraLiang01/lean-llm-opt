LEGACY_OBSERVATION = '"Product Name","Revenue","Demand","Initial Inventory"\n"FDK57","119.144",30,200\n"FDK57","119.144",40,100\n"FDK57","120.144",50,150'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '30', 'Initial Inventory': '200'}}, {'source': '', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '40', 'Initial Inventory': '100'}}, {'source': '', 'values': {'Product Name': 'FDK57', 'Revenue': '120.144', 'Demand': '50', 'Initial Inventory': '150'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
products = []
revenue = {}
demand = {}
inventory = {}
for idx, rec in enumerate(records):
    vals = rec['values']
    key = idx
    products.append(key)
    try:
        revenue[key] = float(vals['Revenue'])
        demand[key] = int(vals['Demand'])
        inventory[key] = int(vals['Initial Inventory'])
    except KeyError as e:
        raise ValueError(f'Missing required field {e} in record {idx}')
for key in products:
    if key not in revenue or key not in demand or key not in inventory:
        raise ValueError(f'Missing data for product {key}')
m = gp.Model('FDK57_RevMax')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= demand[i] for i in products), name='')
m.addConstrs((x[i] <= inventory[i] for i in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')