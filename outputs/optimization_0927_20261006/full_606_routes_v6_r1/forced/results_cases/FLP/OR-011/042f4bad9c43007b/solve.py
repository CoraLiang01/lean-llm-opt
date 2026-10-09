import gurobipy as gp
from gurobipy import GRB
products = ['P1']
revenue = {'P1': 434.74}
demand = {'P1': 8171}
inventory = {'P1': 56450}
for i in products:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f'Missing data for product {i}')
m = gp.Model('Supermarket_id999_Fulfillment')
x_vars = m.addVars(products, lb=0, ub={i: min(inventory[i], demand[i]) for i in products}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')