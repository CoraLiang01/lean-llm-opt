import gurobipy as gp
from gurobipy import GRB
products = ['27in 4K Gaming Monitor', '27in FHD Monitor']
revenue = {'27in 4K Gaming Monitor': 261.2933, '27in FHD Monitor': 52.4965}
demand = {'27in 4K Gaming Monitor': 12474, '27in FHD Monitor': 15057}
inventory = {'27in 4K Gaming Monitor': 62440, '27in FHD Monitor': 75500}
for p in products:
    if p not in revenue or p not in demand or p not in inventory:
        raise ValueError(f'Missing data for product: {p}')
m = gp.Model('27in_Monitor_Revenue_Maximization')
x_vars = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[p] <= demand[p] for p in products), name='')
m.addConstrs((x_vars[p] <= inventory[p] for p in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')