LEGACY_OBSERVATION = '"Product Name","Revenue","Demand","Initial Inventory"\n"FDK57","119.144",30,200\n"FDK57","121.244",30,200\n"FDK57","120.144",50,150\n"FDK57","119.144",40,100\n"FDK57","120.544",10,150\n"FDK57","119.744",50,250'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '30', 'Initial Inventory': '200'}}, {'source': '', 'values': {'Product Name': 'FDK57', 'Revenue': '121.244', 'Demand': '30', 'Initial Inventory': '200'}}, {'source': '', 'values': {'Product Name': 'FDK57', 'Revenue': '120.144', 'Demand': '50', 'Initial Inventory': '150'}}, {'source': '', 'values': {'Product Name': 'FDK57', 'Revenue': '119.144', 'Demand': '40', 'Initial Inventory': '100'}}, {'source': '', 'values': {'Product Name': 'FDK57', 'Revenue': '120.544', 'Demand': '10', 'Initial Inventory': '150'}}, {'source': '', 'values': {'Product Name': 'FDK57', 'Revenue': '119.744', 'Demand': '50', 'Initial Inventory': '250'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
fdk57_records = [rec for rec in records if rec['values'].get('Product Name') == 'FDK57']
n = len(fdk57_records)
revenues = []
demands = []
inventories = []
for rec in fdk57_records:
    v = rec['values']
    try:
        revenues.append(float(v['Revenue']))
        demands.append(int(v['Demand']))
        inventories.append(int(v['Initial Inventory']))
    except KeyError as e:
        raise ValueError(f'Missing required field {e} in record {v}')
if not len(revenues) == len(demands) == len(inventories) == n:
    raise ValueError('Data dimension mismatch in LEGACY_RECORDS for FDK57.')
upper_bounds = [min(demands[i], inventories[i]) for i in range(n)]
m = gp.Model('FDK57_Allocation')
x = m.addVars(n, lb=0, ub=upper_bounds, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenues[i] * x[i] for i in range(n))), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in range(n):
        print(f'x[{i + 1}]: {x[i].X}')
else:
    print(f'Solver status: {m.Status}')