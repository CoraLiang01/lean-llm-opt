LEGACY_OBSERVATION = '{"values": {"Sub Category": "Organic Fruits", "Revenue": "60.8", "Demand": "678906", "Initial Inventory": "5034020.0"}}\n{"values": {"Sub Category": "Organic Staples", "Revenue": "918.45", "Demand": "749927", "Initial Inventory": "5589290.0"}}\n{"values": {"Sub Category": "Organic Vegetables", "Revenue": "77.52", "Demand": "699808", "Initial Inventory": "5202710.0"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Sub Category': 'Organic Fruits', 'Revenue': '60.8', 'Demand': '678906', 'Initial Inventory': '5034020.0'}}, {'source': '', 'values': {'Sub Category': 'Organic Staples', 'Revenue': '918.45', 'Demand': '749927', 'Initial Inventory': '5589290.0'}}, {'source': '', 'values': {'Sub Category': 'Organic Vegetables', 'Revenue': '77.52', 'Demand': '699808', 'Initial Inventory': '5202710.0'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
products = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    vals = rec['values']
    pid = vals['Sub Category']
    products.append(pid)
    try:
        revenue[pid] = float(vals['Revenue'])
        demand[pid] = int(float(vals['Demand']))
        inventory[pid] = int(float(vals['Initial Inventory']))
    except Exception as e:
        raise ValueError(f'Invalid data for {pid}: {e}')
for pid in products:
    if pid not in revenue or pid not in demand or pid not in inventory:
        raise ValueError(f'Missing data for {pid}')
m = gp.Model('Organ_Revenue_Max')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[pid] * x[pid] for pid in products)), GRB.MAXIMIZE)
for pid in products:
    m.addConstr(x[pid] <= demand[pid], name=f'demand_{pid}')
    m.addConstr(x[pid] <= inventory[pid], name=f'inventory_{pid}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')