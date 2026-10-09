import gurobipy as gp
from gurobipy import GRB
items = ['id999']
revenue = {'id999': 434.74}
demand = {'id999': 8171}
inventory = {'id999': 56450}
for i in items:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f'Missing data for item {i}')
m = gp.Model('Supermarket_id999_Allocation')
x_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in items), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in items), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')