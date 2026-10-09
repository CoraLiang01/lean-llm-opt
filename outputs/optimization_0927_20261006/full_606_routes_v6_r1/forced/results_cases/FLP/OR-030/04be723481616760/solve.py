import gurobipy as gp
from gurobipy import GRB
models = ['FDK57_1', 'FDK57_2', 'FDK57_3', 'FDK57_4', 'FDK57_5']
revenue = {'FDK57_1': 119.144, 'FDK57_2': 119.144, 'FDK57_3': 120.144, 'FDK57_4': 121.244, 'FDK57_5': 120.544}
demand = {'FDK57_1': 30, 'FDK57_2': 40, 'FDK57_3': 50, 'FDK57_4': 30, 'FDK57_5': 10}
initial_inventory = {'FDK57_1': 200, 'FDK57_2': 100, 'FDK57_3': 150, 'FDK57_4': 200, 'FDK57_5': 150}
upper_bounds = {}
for i in models:
    if i not in revenue or i not in demand or i not in initial_inventory:
        raise ValueError(f'Missing data for {i}')
    upper_bounds[i] = min(demand[i], initial_inventory[i])
m = gp.Model('FDK57_Revenue_Max')
x_vars = m.addVars(models, lb=0, ub=[upper_bounds[i] for i in models], vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in models)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')