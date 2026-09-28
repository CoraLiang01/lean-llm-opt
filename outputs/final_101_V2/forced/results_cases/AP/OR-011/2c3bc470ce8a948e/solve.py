import gurobipy as gp
from gurobipy import GRB
products = ['id999']
revenue = {'id999': 434.74}
initial_inventory = {'id999': 56450}
demand = {'id999': 8171}
for p in products:
    if p not in revenue or p not in initial_inventory or p not in demand:
        raise ValueError(f'Missing data for product {p}')
m = gp.Model('Supermarket_Fulfillment')
x = m.addVar(vtype=GRB.INTEGER, lb=0, ub=min(initial_inventory['id999'], demand['id999']), name='x999')
m.setObjective(revenue['id999'] * x, GRB.MAXIMIZE)
m.addConstr(x >= 0, name='lb')
m.addConstr(x <= initial_inventory['id999'], name='inv')
m.addConstr(x <= demand['id999'], name='dem')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')