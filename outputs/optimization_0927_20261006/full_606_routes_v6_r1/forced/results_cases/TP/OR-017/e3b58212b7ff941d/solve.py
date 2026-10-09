import gurobipy as gp
from gurobipy import GRB
skus = ['ZZ2AO', 'ZZDW7', 'ZZM1A', 'ZZNC5', 'ZZX6K']
revenue = {'ZZ2AO': 24.38, 'ZZDW7': 30.12, 'ZZM1A': 19.52, 'ZZNC5': 10.79, 'ZZX6K': 111.81}
demand = {'ZZ2AO': 2, 'ZZDW7': 4, 'ZZM1A': 82, 'ZZNC5': 2, 'ZZX6K': 2}
initial_inventory = {'ZZ2AO': 10.0, 'ZZDW7': 20.0, 'ZZM1A': 530.0, 'ZZNC5': 10.0, 'ZZX6K': 10.0}
for sku in skus:
    if sku not in revenue or sku not in demand or sku not in initial_inventory:
        raise ValueError(f'Missing data for SKU {sku}')

def build_model():
    m = gp.Model('ZZ_Revenue_Max')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(skus, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((revenue[sku] * x_vars[sku] for sku in skus)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[sku] <= demand[sku] for sku in skus), name='')
    m.addConstrs((x_vars[sku] <= initial_inventory[sku] for sku in skus), name='')
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')