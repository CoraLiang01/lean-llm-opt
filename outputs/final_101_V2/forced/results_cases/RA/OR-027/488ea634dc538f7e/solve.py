LEGACY_OBSERVATION = '{"values": {"Sub Category": "Organic Fruits", "Revenue": "60.8", "Demand": "678906", "Initial Inventory": "5034020.0"}}\n{"values": {"Sub Category": "Organic Staples", "Revenue": "918.45", "Demand": "749927", "Initial Inventory": "5589290.0"}}\n{"values": {"Sub Category": "Organic Vegetables", "Revenue": "77.52", "Demand": "699808", "Initial Inventory": "5202710.0"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Sub Category': 'Organic Fruits', 'Revenue': '60.8', 'Demand': '678906', 'Initial Inventory': '5034020.0'}}, {'source': '', 'values': {'Sub Category': 'Organic Staples', 'Revenue': '918.45', 'Demand': '749927', 'Initial Inventory': '5589290.0'}}, {'source': '', 'values': {'Sub Category': 'Organic Vegetables', 'Revenue': '77.52', 'Demand': '699808', 'Initial Inventory': '5202710.0'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
products = []
revenue = {}
demand = {}
init_inventory = {}
for rec in records:
    vals = rec['values']
    pid = vals['Sub Category']
    products.append(pid)
    try:
        revenue[pid] = float(vals['Revenue'])
        demand[pid] = int(float(vals['Demand']))
        init_inventory[pid] = float(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data for product {pid}: {e}')
for pid in products:
    if pid not in revenue or pid not in demand or pid not in init_inventory:
        raise ValueError(f'Missing data for product {pid}')
m = gp.Model('Organ_Fulfillment')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
for i in products:
    m.addConstr(x[i] <= demand[i], name=f'demand_{i}')
    m.addConstr(x[i] <= init_inventory[i], name=f'inv_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')