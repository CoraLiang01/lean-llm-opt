import gurobipy as gp
from gurobipy import GRB
models = [1, 2, 3, 4, 5]
revenue = {1: 119.144, 2: 119.144, 3: 120.144, 4: 121.244, 5: 120.544}
initial_inventory = {1: 200, 2: 100, 3: 150, 4: 200, 5: 150}
demand = {1: 30, 2: 40, 3: 50, 4: 30, 5: 10}
for i in models:
    if i not in revenue or i not in initial_inventory or i not in demand:
        raise ValueError(f'Missing data for model {i}')
upper_bound = {i: min(initial_inventory[i], demand[i]) for i in models}
m = gp.Model('Car_Dealership_RevMax')
x = m.addVars(models, vtype=GRB.INTEGER, lb=0, ub=[upper_bound[i] for i in models], name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in models)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in models:
        print(f'x[{i}]: {x[i].X}')
else:
    print(f'Solver status: {m.Status}')