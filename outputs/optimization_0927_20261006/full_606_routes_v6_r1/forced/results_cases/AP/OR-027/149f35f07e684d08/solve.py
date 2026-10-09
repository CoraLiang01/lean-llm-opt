import gurobipy as gp
from gurobipy import GRB
products = [{'Sub Category': 'Organic Fruits', 'Revenue': 60.8, 'Demand': 678906, 'Initial Inventory': 5034020.0}, {'Sub Category': 'Organic Staples', 'Revenue': 918.45, 'Demand': 749927, 'Initial Inventory': 5589290.0}, {'Sub Category': 'Organic Vegetables', 'Revenue': 77.52, 'Demand': 699808, 'Initial Inventory': 5202710.0}]
product_ids = [p['Sub Category'] for p in products]
revenue = {p['Sub Category']: p['Revenue'] for p in products}
demand = {p['Sub Category']: int(p['Demand']) for p in products}
initial_inventory = {p['Sub Category']: int(p['Initial Inventory']) for p in products}
for pid in product_ids:
    if pid not in revenue or pid not in demand or pid not in initial_inventory:
        raise ValueError(f'Missing data for product {pid}')
m = gp.Model('Organ_Revenue_Max')
x_vars = m.addVars(product_ids, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[pid] * x_vars[pid] for pid in product_ids)), GRB.MAXIMIZE)
for pid in product_ids:
    upper = min(initial_inventory[pid], demand[pid])
    m.addConstr(x_vars[pid] >= 0, name=f'lb_{pid}')
    m.addConstr(x_vars[pid] <= upper, name=f'ub_{pid}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')