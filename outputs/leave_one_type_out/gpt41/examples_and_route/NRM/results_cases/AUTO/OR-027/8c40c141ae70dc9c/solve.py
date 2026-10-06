LEGACY_OBSERVATION = '{"values": {"Sub Category": "Organic Fruits", "Revenue": "60.8", "Demand": "678906", "Initial Inventory": "5034020.0"}}\n{"values": {"Sub Category": "Organic Staples", "Revenue": "918.45", "Demand": "749927", "Initial Inventory": "5589290.0"}}\n{"values": {"Sub Category": "Organic Vegetables", "Revenue": "77.52", "Demand": "699808", "Initial Inventory": "5202710.0"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Sub Category': 'Organic Fruits', 'Revenue': '60.8', 'Demand': '678906', 'Initial Inventory': '5034020.0'}}, {'source': '', 'values': {'Sub Category': 'Organic Staples', 'Revenue': '918.45', 'Demand': '749927', 'Initial Inventory': '5589290.0'}}, {'source': '', 'values': {'Sub Category': 'Organic Vegetables', 'Revenue': '77.52', 'Demand': '699808', 'Initial Inventory': '5202710.0'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
categories = []
revenue = {}
demand = {}
init_inventory = {}
for rec in records:
    vals = rec['values']
    cat = vals['Sub Category']
    categories.append(cat)
    try:
        revenue[cat] = float(vals['Revenue'])
        demand[cat] = int(float(vals['Demand']))
        init_inventory[cat] = float(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data for {cat}: {e}')
for cat in categories:
    if cat not in revenue or cat not in demand or cat not in init_inventory:
        raise ValueError(f'Missing data for {cat}')
upper_bounds = {cat: min(demand[cat], init_inventory[cat]) for cat in categories}
m = gp.Model('Organ_Product_Fulfillment')
x = m.addVars(categories, lb=0, ub=upper_bounds, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[cat] * x[cat] for cat in categories)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')