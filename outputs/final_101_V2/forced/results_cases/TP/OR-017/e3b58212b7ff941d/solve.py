import gurobipy as gp
from gurobipy import GRB
skus = ['ZZ2AO', 'ZZDW7', 'ZZM1A', 'ZZNC5', 'ZZX6K']
revenue = {'ZZ2AO': 24.38, 'ZZDW7': 30.12, 'ZZM1A': 19.52, 'ZZNC5': 10.79, 'ZZX6K': 111.81}
demand = {'ZZ2AO': 2, 'ZZDW7': 4, 'ZZM1A': 82, 'ZZNC5': 2, 'ZZX6K': 2}
inventory = {'ZZ2AO': 10.0, 'ZZDW7': 20.0, 'ZZM1A': 530.0, 'ZZNC5': 10.0, 'ZZX6K': 10.0}
for k in skus:
    if k not in revenue or k not in demand or k not in inventory:
        raise ValueError(f'Missing data for SKU {k}')
upper_bounds = {k: min(demand[k], inventory[k]) for k in skus}
m = gp.Model('ZZ_Revenue_Max')
x = m.addVars(skus, lb=0, ub=[upper_bounds[k] for k in skus], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[k] * x[k] for k in skus)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for k in skus:
        print(f'{x[k].VarName}: {x[k].X}')
else:
    print(f'Solver status: {m.Status}')