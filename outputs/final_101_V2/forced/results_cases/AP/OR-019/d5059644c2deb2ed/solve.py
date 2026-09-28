import gurobipy as gp
from gurobipy import GRB
products = [{'Product Name': '27in 4K Gaming Monitor', 'Revenue': 389.99, 'Initial Inventory': 62440, 'Demand': 12474}, {'Product Name': '27in FHD Monitor', 'Revenue': 149.99, 'Initial Inventory': 75500, 'Demand': 15057}]
product_keys = ['27in 4K Gaming Monitor', '27in FHD Monitor']
revenue = {}
inventory = {}
demand = {}
for p in products:
    key = p['Product Name']
    revenue[key] = p['Revenue']
    inventory[key] = p['Initial Inventory']
    demand[key] = p['Demand']
for key in product_keys:
    if key not in revenue or key not in inventory or key not in demand:
        raise ValueError(f'Missing data for product: {key}')
upper_bounds = {key: min(inventory[key], demand[key]) for key in product_keys}
m = gp.Model('27in_Monitor_Revenue_Max')
x = m.addVars(product_keys, vtype=GRB.INTEGER, lb=0, ub=[upper_bounds[k] for k in product_keys], name='')
m.setObjective(gp.quicksum((revenue[k] * x[k] for k in product_keys)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')