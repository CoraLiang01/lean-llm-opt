import gurobipy as gp
from gurobipy import GRB
models = [1, 2, 3, 4, 5, 6, 7]
revenue = {1: 119.144, 2: 119.144, 3: 120.144, 4: 119.744, 5: 120.844, 6: 121.244, 7: 121.244}
demand = {1: 30, 2: 40, 3: 50, 4: 50, 5: 50, 6: 30, 7: 30}
initial_inventory = {1: 200, 2: 100, 3: 150, 4: 250, 5: 150, 6: 150, 7: 200}
for i in models:
    if i not in revenue or i not in demand or i not in initial_inventory:
        raise ValueError(f'Missing data for model {i}')
m = gp.Model('Car_Dealership_FDK57')
x_vars = m.addVars(models, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in models)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in models), name='')
m.addConstrs((x_vars[i] <= initial_inventory[i] for i in models), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')