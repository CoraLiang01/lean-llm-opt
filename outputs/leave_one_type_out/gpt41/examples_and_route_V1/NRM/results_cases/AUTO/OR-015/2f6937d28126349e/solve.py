LEGACY_OBSERVATION = 'Product Name,Revenue,Initial Inventory,Demand\nAalopuri,20,10440.0,1483'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': 'Aalopuri', 'Revenue': '20', 'Initial Inventory': '10440.0', 'Demand': '1483'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
aalopuri_data = None
for rec in records:
    values = rec['values']
    if values.get('Product Name') == 'Aalopuri':
        aalopuri_data = values
        break
if aalopuri_data is None:
    raise ValueError('Aalopuri data not found in LEGACY_RECORDS.')
try:
    revenue = float(aalopuri_data['Revenue'])
    initial_inventory = float(aalopuri_data['Initial Inventory'])
    demand = int(float(aalopuri_data['Demand']))
except Exception as e:
    raise ValueError(f'Error parsing Aalopuri data: {e}')
m = gp.Model('Aalopuri_Fulfillment')
x = m.addVar(lb=0, vtype=GRB.INTEGER, name='x_Aalopuri')
m.setObjective(revenue * x, GRB.MAXIMIZE)
m.addConstr(x <= initial_inventory, name='inv')
m.addConstr(x <= demand, name='dem')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')