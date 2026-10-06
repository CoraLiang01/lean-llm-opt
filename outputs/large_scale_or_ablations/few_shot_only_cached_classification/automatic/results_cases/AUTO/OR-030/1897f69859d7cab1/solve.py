import gurobipy as gp
from gurobipy import GRB
products = [1, 2, 3, 4]
revenue = {1: 119.144, 2: 121.244, 3: 120.544, 4: 120.844}
demand = {1: 30, 2: 30, 3: 10, 4: 50}
inventory = {1: 200, 2: 200, 3: 150, 4: 150}
for i in products:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f'Missing data for product index {i}')
m = gp.Model('FDK57_RevMax')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= inventory[i] for i in products), name='')
m.addConstrs((x[i] <= demand[i] for i in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')