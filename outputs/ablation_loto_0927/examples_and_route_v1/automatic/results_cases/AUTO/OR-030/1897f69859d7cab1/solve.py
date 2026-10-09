LEGACY_OBSERVATION = '{"values": {"Product Name": "FDK57", "Revenue": "119.144", "Demand": "30", "Initial Inventory": "200"}}\n{"values": {"Product Name": "FDK57", "Revenue": "119.144", "Demand": "40", "Initial Inventory": "100"}}\n{"values": {"Product Name": "FDK57", "Revenue": "120.144", "Demand": "50", "Initial Inventory": "150"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '30', 'Initial Inventory': '200'}}, {'source': '', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '40', 'Initial Inventory': '100'}}, {'source': '', 'values': {'Product Name': 'FDK57', 'Revenue': '120.144', 'Demand': '50', 'Initial Inventory': '150'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
products = []
revenue = {}
demand = {}
inventory = {}
for (idx, rec) in enumerate(records):
    vals = rec['values']
    key = f'FDK57_{idx + 1}'
    products.append(key)
    try:
        revenue[key] = float(vals['Revenue'])
        demand[key] = int(vals['Demand'])
        inventory[key] = int(vals['Initial Inventory'])
    except KeyError as e:
        raise ValueError(f'Missing data for {key}: {e}')
if not set(revenue.keys()) == set(products) == set(demand.keys()) == set(inventory.keys()):
    raise ValueError('Coefficient dimensions or identifiers do not match decision index set.')
m = gp.Model('FDK57_Allocation')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= demand[i] for i in products), name='')
m.addConstrs((x[i] <= inventory[i] for i in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')