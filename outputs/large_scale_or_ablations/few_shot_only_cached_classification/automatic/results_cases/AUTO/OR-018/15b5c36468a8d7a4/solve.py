import gurobipy as gp
from gurobipy import GRB
items = ['Baby Food_255.28']
revenue = {'Baby Food_255.28': 255.28}
demand = {'Baby Food_255.28': 3066513}
inventory = {'Baby Food_255.28': 22749210}
for i in items:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f'Missing data for item: {i}')
m = gp.Model('Baby_Food_Revenue_Max')
x1 = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='x1')
m.setObjective(255.28 * x1, GRB.MAXIMIZE)
m.addConstr(x1 <= 3066513, name='demand')
m.addConstr(x1 <= 22749210, name='inventory')
m.addConstr(x1 >= 0, name='nonneg')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')