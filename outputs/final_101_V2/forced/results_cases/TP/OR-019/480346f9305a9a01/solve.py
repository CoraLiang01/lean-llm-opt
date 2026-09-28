import gurobipy as gp
from gurobipy import GRB
P = {1: '27in 4K Gaming Monitor', 2: '27in FHD Monitor'}
revenue = {1: 389.99, 2: 149.99}
demand = {1: 12474, 2: 15057}
initial_inventory = {1: 62440, 2: 75500}
for i in P:
    if i not in revenue or i not in demand or i not in initial_inventory:
        raise ValueError(f'Missing data for product {i}')
bounds = {i: min(demand[i], initial_inventory[i]) for i in P}
m = gp.Model('27in_Product_Revenue')
x = m.addVars(P.keys(), lb=0, ub=bounds, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in P)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')