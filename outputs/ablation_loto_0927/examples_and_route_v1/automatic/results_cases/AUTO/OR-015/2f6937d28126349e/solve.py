LEGACY_OBSERVATION = '{"values": {"Product Name": "Aalopuri", "Revenue": "20", "Demand": "1483", "Initial Inventory": "10440.0"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': 'Aalopuri', 'Revenue': '20', 'Demand': '1483', 'Initial Inventory': '10440.0'}}]
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
    revenue = int(aalopuri_data['Revenue'])
    demand = int(aalopuri_data['Demand'])
    initial_inventory = int(float(aalopuri_data['Initial Inventory']))
except Exception as e:
    raise ValueError(f'Error parsing coefficients: {e}')
upper_bound = min(demand, initial_inventory)
m = gp.Model('Aalopuri_Fulfillment')
x = m.addVar(lb=0, ub=upper_bound, vtype=GRB.INTEGER, name='x_Aalopuri')
m.setObjective(revenue * x, GRB.MAXIMIZE)
m.addConstr(x <= demand, name='demand')
m.addConstr(x <= initial_inventory, name='inventory')
m.addConstr(x >= 0, name='nonneg')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')