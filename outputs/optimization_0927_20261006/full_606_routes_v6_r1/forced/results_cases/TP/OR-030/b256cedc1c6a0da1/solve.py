import gurobipy as gp
from gurobipy import GRB
models = [1, 2, 3, 4, 5, 6]
revenue = {1: 119.144, 2: 119.144, 3: 120.144, 4: 121.244, 5: 120.544, 6: 120.844}
demand = {1: 30, 2: 40, 3: 50, 4: 30, 5: 10, 6: 50}
initial_inventory = {1: 200, 2: 100, 3: 150, 4: 200, 5: 150, 6: 150}
upper_bounds = {i: min(demand[i], initial_inventory[i]) for i in models}
for i in models:
    if i not in revenue or i not in demand or i not in initial_inventory:
        raise ValueError(f'Missing data for model {i}')
m = gp.Model('FDK57_RevMax')
x_vars = m.addVars(models, lb=0, ub=[upper_bounds[i] for i in models], vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in models)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')