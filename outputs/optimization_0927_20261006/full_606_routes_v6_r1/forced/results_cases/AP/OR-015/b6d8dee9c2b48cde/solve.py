import gurobipy as gp
from gurobipy import GRB
aalop_products = ['Aalopuri']
revenue = {'Aalopuri': 20}
demand = {'Aalopuri': 1483}
initial_inventory = {'Aalopuri': 10440.0}
for i in aalop_products:
    if i not in revenue or i not in demand or i not in initial_inventory:
        raise ValueError(f'Missing data for product {i}')
m = gp.Model('Aalop_Revenue_Max')
x_vars = m.addVars(aalop_products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in aalop_products)), GRB.MAXIMIZE)
for i in aalop_products:
    m.addConstr(x_vars[i] <= min(initial_inventory[i], demand[i]), name=f'ub_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')