import gurobipy as gp
from gurobipy import GRB
workstations = [1, 2, 3]
models = [f'HiFi{k}' for k in range(1, 102)]
C = {1: 1296.0, 2: 1238.4, 3: 1267.2}
t_1 = [6, 4, 6, 7, 6, 6, 8, 9, 6, 7, 1, 2, 4, 7, 3, 8, 3, 2, 4, 5, 8, 3, 2, 3, 9, 7, 3, 5, 7, 6, 2, 1, 5, 6, 5, 1, 7, 9, 8, 3, 3, 8, 2, 3, 3, 8, 9, 2, 3, 4, 2, 9, 2, 1, 8, 8, 4, 4, 6, 1, 6, 5, 3, 5, 1, 6, 6, 5, 3, 4, 3, 8, 1, 2, 3, 2, 8, 4, 4, 2, 7, 5, 1, 6, 4, 1, 3, 8, 3, 3, 3, 3, 6, 7, 6, 2, 1, 8, 9, 7, 9]
t_2 = [5, 5, 5, 1, 7, 8, 7, 5, 6, 8, 9, 9, 2, 6, 9, 4, 1, 2, 9, 3, 8, 5, 9, 5, 8, 7, 1, 1, 9, 7, 1, 9, 6, 4, 7, 4, 8, 6, 5, 3, 6, 7, 6, 2, 1, 1, 3, 8, 4, 3, 6, 9, 8, 7, 2, 2, 5, 4, 3, 8, 8, 6, 6, 3, 1, 6, 2, 6, 1, 3, 7, 1, 1, 2, 8, 7, 8, 8, 7, 5, 2, 5, 6, 2, 3, 2, 3, 8, 4, 9, 6, 1, 4, 8, 8, 6, 8, 5, 5, 8, 3]
t_3 = [4, 6, 5, 2, 6, 5, 3, 3, 4, 8, 6, 3, 3, 3, 7, 8, 3, 8, 1, 5, 3, 8, 5, 8, 4, 8, 6, 7, 9, 5, 3, 6, 3, 3, 3, 8, 4, 6, 3, 8, 3, 7, 5, 3, 1, 8, 9, 6, 6, 4, 7, 1, 9, 9, 3, 9, 6, 5, 7, 8, 9, 9, 8, 5, 4, 4, 3, 3, 8, 8, 2, 4, 9, 6, 7, 6, 7, 3, 1, 7, 6, 4, 3, 5, 7, 6, 3, 5, 2, 2, 9, 3, 6, 9, 7, 2, 4, 5, 8, 1, 6]
if not len(t_1) == len(t_2) == len(t_3) == len(models):
    raise ValueError('Processing time lists must match number of models.')
t = {1: {models[k]: t_1[k] for k in range(101)}, 2: {models[k]: t_2[k] for k in range(101)}, 3: {models[k]: t_3[k] for k in range(101)}}
m = gp.Model('HiFi_IdleTime_Min')
x_vars = m.addVars(models, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(C[1] - gp.quicksum((t[1][k] * x_vars[k] for k in models)) + (C[2] - gp.quicksum((t[2][k] * x_vars[k] for k in models))) + (C[3] - gp.quicksum((t[3][k] * x_vars[k] for k in models))), GRB.MINIMIZE)
m.addConstr(gp.quicksum((t[1][k] * x_vars[k] for k in models)) <= C[1], name='cap1')
m.addConstr(gp.quicksum((t[2][k] * x_vars[k] for k in models)) <= C[2], name='cap2')
m.addConstr(gp.quicksum((t[3][k] * x_vars[k] for k in models)) <= C[3], name='cap3')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')