import gurobipy as gp
from gurobipy import GRB
products = ['id999']
revenue = {'id999': 434.74}
demand = {'id999': 8171}
inventory = {'id999': 56450}
for i in products:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f'Missing data for product {i}')
m = gp.Model('Supermarket_id999_Allocation')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= demand[i] for i in products), name='')
m.addConstrs((x[i] <= inventory[i] for i in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')