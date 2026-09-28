LEGACY_OBSERVATION = '```csv\nProduct Name,Revenue,Initial Inventory,Demand\nFDK57,119.144,200,30\nFDK57,120.144,150,50\nFDK57,121.244,150,30\n```'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Initial Inventory': '200', 'Demand': '30'}}, {'source': '', 'values': {'Product Name': 'FDK57', 'Revenue': '120.144', 'Initial Inventory': '150', 'Demand': '50'}}, {'source': '', 'values': {'Product Name': 'FDK57', 'Revenue': '121.244', 'Initial Inventory': '150', 'Demand': '30'}}]
import gurobipy as gp
from gurobipy import GRB
records = [rec for rec in LEGACY_RECORDS if rec['values'].get('Product Name') == 'FDK57']
if len(records) == 0:
    raise ValueError('No FDK57 records found in LEGACY_RECORDS.')
n = len(records)
revenue = {}
inventory = {}
demand = {}
for idx, rec in enumerate(records):
    key = idx + 1
    vals = rec['values']
    try:
        revenue[key] = float(vals['Revenue'])
        inventory[key] = int(vals['Initial Inventory'])
        demand[key] = int(vals['Demand'])
    except KeyError as e:
        raise ValueError(f'Missing field {e} in record {idx}')
m = gp.Model('FDK57_Allocation')
x = m.addVars(range(1, n + 1), lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in range(1, n + 1))), GRB.MAXIMIZE)
for i in range(1, n + 1):
    upper = min(inventory[i], demand[i])
    m.addConstr(x[i] <= upper, name=f'cap_{i}')
    m.addConstr(x[i] >= 0, name=f'nonneg_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')