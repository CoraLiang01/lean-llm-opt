import gurobipy as gp
from gurobipy import GRB
models = ['FDK57_1', 'FDK57_2', 'FDK57_3']
revenue = {'FDK57_1': 119.144, 'FDK57_2': 119.144, 'FDK57_3': 120.144}
demand = {'FDK57_1': 30, 'FDK57_2': 40, 'FDK57_3': 50}
initial_inventory = {'FDK57_1': 200, 'FDK57_2': 100, 'FDK57_3': 150}
for k in models:
    if k not in revenue or k not in demand or k not in initial_inventory:
        raise ValueError(f'Missing data for model {k}')
m = gp.Model('FDK57_RevMax')
x = m.addVars(models, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[k] * x[k] for k in models)), GRB.MAXIMIZE)
for k in models:
    m.addConstr(x[k] <= demand[k], name=f'demand_{k}')
    m.addConstr(x[k] <= initial_inventory[k], name=f'inventory_{k}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')