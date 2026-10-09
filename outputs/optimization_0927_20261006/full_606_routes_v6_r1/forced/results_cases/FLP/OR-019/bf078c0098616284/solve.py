import gurobipy as gp
from gurobipy import GRB
products = {1: '27in 4K Gaming Monitor', 2: '27in FHD Monitor'}
revenue = {1: 389.99, 2: 149.99}
demand = {1: 12474, 2: 15057}
inventory = {1: 62440, 2: 75500}
for i in products:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f'Missing data for product {i}: {products.get(i, str(i))}')
m = gp.Model('27in_Product_Revenue_Maximization')
x_vars = m.addVars(products.keys(), lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')