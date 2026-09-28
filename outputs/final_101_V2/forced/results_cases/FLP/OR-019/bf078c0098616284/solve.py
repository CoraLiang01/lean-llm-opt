import gurobipy as gp
from gurobipy import GRB
products = ['27in 4K Gaming Monitor', '27in FHD Monitor']
revenue = {'27in 4K Gaming Monitor': 389.99, '27in FHD Monitor': 149.99}
inventory = {'27in 4K Gaming Monitor': 62440, '27in FHD Monitor': 75500}
demand = {'27in 4K Gaming Monitor': 12474, '27in FHD Monitor': 15057}
for p in products:
    if p not in revenue or p not in inventory or p not in demand:
        raise ValueError(f'Missing data for product: {p}')
m = gp.Model('27in_Product_Revenue_Max')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= inventory[i] for i in products), name='')
m.addConstrs((x[i] <= demand[i] for i in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')