import gurobipy as gp
from gurobipy import GRB
products = [{'id_number': 'id999', 'Revenue': 434.74, 'Demand': 8171, 'Initial Inventory': 56450}]
product_ids = [p['id_number'] for p in products]
revenue = {p['id_number']: p['Revenue'] for p in products}
demand = {p['id_number']: p['Demand'] for p in products}
initial_inventory = {p['id_number']: p['Initial Inventory'] for p in products}
for pid in product_ids:
    if pid not in revenue or pid not in demand or pid not in initial_inventory:
        raise ValueError(f'Missing data for product {pid}')
m = gp.Model('Supermarket_Revenue_Maximization')
x_vars = m.addVars(product_ids, vtype=GRB.INTEGER, lb=0, name='')
for pid in product_ids:
    upper_bound = min(initial_inventory[pid], demand[pid])
    m.addConstr(x_vars[pid] <= upper_bound, name=f'ub_{pid}')
    m.addConstr(x_vars[pid] >= 0, name=f'lb_{pid}')
m.setObjective(gp.quicksum((revenue[pid] * x_vars[pid] for pid in product_ids)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')