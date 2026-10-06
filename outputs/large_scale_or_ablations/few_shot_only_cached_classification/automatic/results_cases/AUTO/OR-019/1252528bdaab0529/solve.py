import gurobipy as gp
from gurobipy import GRB
items = ['27in 4K Gaming Monitor', '27in FHD Monitor']
revenue = {'27in 4K Gaming Monitor': 389.99, '27in FHD Monitor': 149.99}
demand = {'27in 4K Gaming Monitor': 12474, '27in FHD Monitor': 15057}
inventory = {'27in 4K Gaming Monitor': 62440, '27in FHD Monitor': 75500}
for i in items:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f'Missing data for item: {i}')
m = gp.Model('27in_Monitor_Revenue_Max')
x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
m.addConstr(x['27in 4K Gaming Monitor'] <= demand['27in 4K Gaming Monitor'], name='c1')
m.addConstr(x['27in 4K Gaming Monitor'] <= inventory['27in 4K Gaming Monitor'], name='c2')
m.addConstr(x['27in FHD Monitor'] <= demand['27in FHD Monitor'], name='c3')
m.addConstr(x['27in FHD Monitor'] <= inventory['27in FHD Monitor'], name='c4')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')