import gurobipy as gp
from gurobipy import GRB
products = {1: '27in 4K Gaming Monitor', 2: '27in FHD Monitor'}
revenue = {1: 261.2933, 2: 52.4965}
demand = {1: 12474, 2: 15057}
inventory = {1: 62440, 2: 75500}
for i in products:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f'Missing data for product index {i}')
m = gp.Model('27in_Product_Revenue_Maximization')
x = m.addVars([1, 2], lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(revenue[1] * x[1] + revenue[2] * x[2], GRB.MAXIMIZE)
m.addConstr(x[1] <= demand[1], name='demand1')
m.addConstr(x[1] <= inventory[1], name='inventory1')
m.addConstr(x[2] <= demand[2], name='demand2')
m.addConstr(x[2] <= inventory[2], name='inventory2')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')