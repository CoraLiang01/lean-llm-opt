import gurobipy as gp
from gurobipy import GRB
products = ['Aalopuri']
revenue = {'Aalopuri': 20}
initial_inventory = {'Aalopuri': 10440.0}
demand = {'Aalopuri': 1483}
for p in products:
    if p not in revenue or p not in initial_inventory or p not in demand:
        raise ValueError(f'Missing data for product {p}')
m = gp.Model('Aalopuri_Inventory')
x = m.addVar(vtype=GRB.INTEGER, name='xA')
m.setObjective(revenue['Aalopuri'] * x, GRB.MAXIMIZE)
m.addConstr(x <= initial_inventory['Aalopuri'], name='inv')
m.addConstr(x <= demand['Aalopuri'], name='dem')
m.addConstr(x >= 0, name='nonneg')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')