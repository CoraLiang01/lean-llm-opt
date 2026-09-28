import gurobipy as gp
from gurobipy import GRB
products = [{'Product Name': '27in 4K Gaming Monitor', 'Revenue': 261.2933, 'Initial Inventory': 62440, 'Demand': 12474}, {'Product Name': '27in FHD Monitor', 'Revenue': 52.4965, 'Initial Inventory': 75500, 'Demand': 15057}]
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
m = gp.Model('DeptStore_27in_Monitor_Revenue')
x = m.addVars(product_keys, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in product_keys)), GRB.MAXIMIZE)
for p in product_keys:
    m.addConstr(x[p] <= inventory[p], name=f'inv_{p}')
for p in product_keys:
    m.addConstr(x[p] <= demand[p], name=f'dem_{p}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')