import gurobipy as gp
from gurobipy import GRB
items = ['Baby Food_255.28']
revenue = {'Baby Food_255.28': 255.28}
demand = {'Baby Food_255.28': 765850}
inventory = {'Baby Food_255.28': 5627060}
for i in items:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f'Missing data for item: {i}')
m = gp.Model('Baby_Food_Revenue_Max')
x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= demand[i] for i in items), name='')
m.addConstrs((x[i] <= inventory[i] for i in items), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')