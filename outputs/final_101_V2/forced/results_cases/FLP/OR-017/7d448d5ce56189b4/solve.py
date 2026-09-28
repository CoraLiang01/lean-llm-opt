import gurobipy as gp
from gurobipy import GRB
I = ['4ZZWJ', 'ZZZTA', 'ZZ2AO', 'ZZSZDW', 'ZZX6K']
revenue = {'4ZZWJ': 8.56, 'ZZZTA': 1.58, 'ZZ2AO': 24.38, 'ZZSZDW': 110.7, 'ZZX6K': 111.81}
inventory = {'4ZZWJ': 10.0, 'ZZZTA': 10.0, 'ZZ2AO': 10.0, 'ZZSZDW': 30.0, 'ZZX6K': 10.0}
demand = {'4ZZWJ': 2, 'ZZZTA': 2, 'ZZ2AO': 2, 'ZZSZDW': 5, 'ZZX6K': 2}
for i in I:
    if i not in revenue or i not in inventory or i not in demand:
        raise ValueError(f'Missing data for product {i}')
m = gp.Model('ZZ_Product_Fulfillment')
x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
for i in I:
    m.addConstr(x[i] <= inventory[i], name=f'inv_{i}')
    m.addConstr(x[i] <= demand[i], name=f'dem_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')