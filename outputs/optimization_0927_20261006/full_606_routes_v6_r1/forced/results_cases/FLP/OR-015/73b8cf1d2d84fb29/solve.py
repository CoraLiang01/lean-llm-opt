import gurobipy as gp
from gurobipy import GRB
products = ['Aalopuri']
revenue = {'Aalopuri': 20}
demand = {'Aalopuri': 1483}
initial_inventory = {'Aalopuri': 10440.0}
for i in products:
    if i not in revenue or i not in demand or i not in initial_inventory:
        raise ValueError(f'Missing data for product {i}')
m = gp.Model('Aalop_Revenue_Maximization')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= initial_inventory[i] for i in products), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')