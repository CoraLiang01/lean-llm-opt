import gurobipy as gp
from gurobipy import GRB
items = ['ZZ2AO', 'ZZDW7', 'ZZM1A', 'ZZNC5', 'ZZX6K']
revenue = {'ZZ2AO': 24.38, 'ZZDW7': 30.12, 'ZZM1A': 19.52, 'ZZNC5': 10.79, 'ZZX6K': 111.81}
demand = {'ZZ2AO': 2, 'ZZDW7': 4, 'ZZM1A': 82, 'ZZNC5': 2, 'ZZX6K': 2}
inventory = {'ZZ2AO': 10.0, 'ZZDW7': 20.0, 'ZZM1A': 530.0, 'ZZNC5': 10.0, 'ZZX6K': 10.0}
for i in items:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f'Missing data for SKU {i}')
m = gp.Model('ZZ_SKU_Revenue_Max')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in items), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in items), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')