import gurobipy as gp
from gurobipy import GRB
products = ['id999']
revenue = {'id999': 434.74}
initial_inventory = {'id999': 56450}
demand = {'id999': 8171}
for i in products:
    if i not in revenue or i not in initial_inventory or i not in demand:
        raise ValueError(f'Missing data for product {i}')
m = gp.Model('Supermarket_id999')
x = m.addVar(lb=0, vtype=GRB.INTEGER, name='x_id999')
m.setObjective(revenue['id999'] * x, GRB.MAXIMIZE)
m.addConstr(x <= initial_inventory['id999'], name='inv')
m.addConstr(x <= demand['id999'], name='dem')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')